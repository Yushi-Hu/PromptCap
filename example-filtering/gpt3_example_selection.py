from multiprocessing import context
import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
import random
import openai
from tqdm import tqdm
import sys
from collections import defaultdict
from torchmetrics import CharErrorRate


def codex(prompt):
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(
                engine="code-davinci-002", 
                prompt=prompt,
                max_tokens=20,
                logprobs=1,
                temperature=0.,
                stream=False,
                stop=["\n", "<|endoftext|>"])
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: 
                # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(5)
    return response


## initial cleaning for reference QA results; Please use vqav2 eval script for the final number
def process_answer(answer):
    answer = answer.lower()
    answer = answer.replace('.','').replace(',','').lower()
    to_be_removed = {'a', 'an', 'the','to', ''}
    answer_list = answer.split(' ')
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return ' '.join(answer_list)

# return the majority answer from a list
def answer_majority(answers):
    vote_dict = defaultdict(int)
    for a in answers:
        vote_dict[a] += 1
    sorted_answers = [k for k, v in sorted(
        vote_dict.items(), key=lambda item: item[1])]
    return sorted_answers[-1]

def get_answers(instance, dataset_type):
    answers = []
    if dataset_type == "vqa2":
        answers = [s['answer'] for s in instance['answers']]
    elif dataset_type == "okvqa":
        answers = [s['answer'] for s in instance['answers']]
    elif dataset_type == "aokvqa":
        answers = instance["direct_answers"]
    else:
        raise ValueError("dataset not supported")
    return answers


def load_anno(dataset_fn, dataset_type):
    with open(dataset_fn) as f:
        vqa_dataset = json.load(f)
    if dataset_type != "aokvqa":
        vqa_dataset = vqa_dataset["annotations"]
    answer_dict = {}
    question_dict = {}
    for sample in vqa_dataset:
        question_id, image_id = sample['question_id'], sample['image_id']
        context_key = f"{image_id}<->{question_id}"
        answers = get_answers(sample, dataset_type)
        question = sample["question"]
        answer_dict[context_key] = answers
        question_dict[context_key] = question
    return answer_dict, question_dict


def load_captions(caption_fn, coco_key="coco_captions", prompt_key = "prompt_guided_captions"):
    with open(caption_fn) as f:
        caption_dataset = json.load(f)
    caption_dict = {}
    
    for sample in caption_dataset:
        question_id, image_id = sample['question_id'], sample['image_id']
        context_key = f"{image_id}<->{question_id}"
        captions = sample[coco_key] 
        prompt_captions =[c for c in sample[prompt_key] if c != ""]
        prompt_captions = list(set(prompt_captions))
        captions.extend(prompt_captions)
        
        # if want to reduce workload, can sample fewer captions here!
        # e.g. captions = captions[:10]
        
        caption_dict[context_key] = captions
    return caption_dict



class Example_selection:
    def __init__(self, args):
        self.args = args

        # load CER evaluator
        self.evaluator = CharErrorRate()

        # load captions
        self.train_captions = load_captions(args.train_caption_file)
        self.val_captions = load_captions(args.val_caption_file)

        # loading input questions (and answer for reference accuracy computing)
        self.answer_dict, self.question_dict = load_anno(args.val_dataset, args.dataset_type)
        self.val_keys = list(self.question_dict.keys())

        self.train_answer_dict,self.train_question_dict = load_anno(args.train_dataset, args.dataset_type)
        self.train_keys = list(self.train_answer_dict.keys())
        
        # load tags
        self.tags_dict = self.load_tags()
        
        self.load_similarity()


    def inference(self):
        chosen_caption_dataset = []

        for key in tqdm(self.val_keys):
            question, answers = self.question_dict[key], self.answer_dict[key]
            answer = answer_majority(answers)
            image_id = int(key.split('<->')[0])
            question_id = key.split('<->')[1]

            captions, soft_scores, hard_scores = self.sample_inference(key)

            this_sample = { "question_id": question_id,
                            "image_id": image_id,
                            "question": question,
                            "answer": answer,
                            "coco_captions": captions[:5],
                            "prompt_guided_captions": captions[5:],
                            "soft_scores": soft_scores,
                            "hard_scores": hard_scores
                            }
            chosen_caption_dataset.append(this_sample)

        return chosen_caption_dataset


    def sample_inference(self, key):
        question, answers = self.question_dict[key], self.answer_dict[key]
        candidate_captions = self.val_captions[key]

        val_tag = self.tags_dict[int(key.split('<->')[0])]

        # in-context examples selection. put most similar at last
        context_key_list = self.get_context_keys(key, \
            self.args.similarity_metric, self.args.n_shot)[::-1]

        # make the prompt
        prompt = 'Please answer the question according to the above context.\n===\n'

        for ni in range(self.args.n_shot):
            if context_key_list is None:
                context_key = self.train_keys[random.randint(0,len(self.train_keys)-1)]
            else:
                context_key = context_key_list[ni]
            
            this_caption = self.train_captions[context_key][5]  # skip 5 coco captions
            this_tag = self.tags_dict[int(context_key.split('<->')[0])]
            this_question = self.train_question_dict[context_key]
            this_answer = answer_majority(self.train_answer_dict[context_key])

            prompt += f"Context: {this_caption} {this_tag}\n===\n"
            prompt += f"Q: {this_question}\nA: {this_answer}\n\n===\n"

        candidate_prompts = []

        for candidate_caption in candidate_captions:
            candidate_prompt = prompt
            candidate_prompt += f"Context: {candidate_caption} {val_tag}\n===\n"
            candidate_prompt += f"Q: {question}\nA:"
            candidate_prompts.append(candidate_prompt)
        
        response = codex(candidate_prompts)

        gpt3_answers = [process_answer(sample["text"]) for sample in response["choices"]]

        # score each answer with CER
        gpt3_answer_scores = []
        hard_scores = []

        for gpt3_answer in gpt3_answers:
            this_answer_scores = []
            count = 0
            for gold_answer in answers:
                this_score = 1 - self.evaluator(gpt3_answer, gold_answer).item()
                this_answer_scores.append(this_score)

                if gpt3_answer == gold_answer:
                    count += 1
            
            gpt3_answer_scores.append(min(1., sum(sorted(this_answer_scores)[-3:])/3))
            hard_scores.append(min(1., count*0.3))
        
        # max_gpt3_answer_score = max(gpt3_answer_scores)
        # max_hard_score = max(hard_scores)

        # chosen_captions = []

        # for caption, caption_score in zip(candidate_captions, gpt3_answer_scores):
        #     if caption_score == max_gpt3_answer_score:
        #         chosen_captions.append(caption)

        # # sort the captions by length, only keep unique ones
        # chosen_captions = list(set(chosen_captions))
        # chosen_captions = sorted(chosen_captions, key=lambda x: len(x))
        
        return candidate_captions, gpt3_answer_scores, hard_scores


    def get_context_keys(self,key,metric,n):
        chosen_index = []
        lineid = self.valkey2idx[key]
        similarity = np.matmul(self.train_feature,self.val_feature[lineid,:])
        chosen_index = similarity.argsort()[-n-1:][::-1]
        
        # if selected in training set, remove the original example
        if self.args.partition == "train":
            chosen_index = chosen_index[1:]
        else:
            chosen_index = chosen_index[:-1]

        return [self.train_idx[str(x)] for x in chosen_index]


    def load_similarity(self):

        embed_key = ""
        if self.args.similarity_metric == "question":
            embed_key = "question"
        elif self.args.similarity_metric == "imagequestion":
            embed_key = "question_image"
        else:
            raise ValueError("Unknown similarity metric")

        if self.args.partition == "val":
            val_idx = json.load(open(f"{self.args.similarity_path}/{self.args.dataset_type}_val_ids.json"))
        else:
            val_idx = json.load(open(f"{self.args.similarity_path}/{self.args.dataset_type}_ids.json"))

        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)

        self.train_feature = np.load(f"{self.args.similarity_path}/{self.args.dataset_type}_{embed_key}.npy")
        self.train_idx = json.load(open(f"{self.args.similarity_path}/{self.args.dataset_type}_ids.json"))
        if self.args.partition == "val":
            self.val_feature = np.load(f"{self.args.similarity_path}/{self.args.dataset_type}_val_{embed_key}.npy")
        else:
            self.val_feature = np.load(f"{self.args.similarity_path}/{self.args.dataset_type}_{embed_key}.npy")
        

    def load_tags(self):
        tags_dict = {}
        tagging_pred_file = '%s/test.score.json.tsv'%self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file,'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]),json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        tagging_pred_file = '%s/val.score.json.tsv'%self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file,'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]),json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        tagging_pred_file = '%s/train.score.json.tsv'%self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file,'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]),json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        return tags_dict



def main():
    parser = argparse.ArgumentParser()

    # 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     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

    parser.add_argument('--apikey', type=str, default="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", help='api key; https://openai.com/api/')
    parser.add_argument('--dataset_type', type=str, default='aokvqa', choices=['okvqa', 'vqa2','aokvqa'])
    parser.add_argument('--partition', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--caption_type', type=str, default='caption_tag', help='caption_tag, caption')
    parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--train_caption_file', type=str, default='../data/raw_captions/aokvqa_w_gpt3_caption_code002.json')
    parser.add_argument('--val_caption_file', type=str, default='../data/raw_captions/aokvqa_w_gpt3_caption_val_code002.json')
    parser.add_argument('--train_dataset', type=str, default='../data/task_data/aokvqa_w_coco_caption.json')
    parser.add_argument('--val_dataset', type=str, default='../data/task_data/aokvqa_w_coco_caption_val.json')
    parser.add_argument('--tag_path', type=str, default='../coco_caption_pred_tags')
    parser.add_argument('--similarity_path', type=str, default='../coco_clip')
    parser.add_argument('--output_fn', default="aokvqa_w_gpt3_caption_val_code002_filtered.json")

    args = parser.parse_args()

    openai.api_key = args.apikey

    caption_selection = Example_selection(args)

    chosen_caption_dataset = caption_selection.inference()

    # sample_count, total_soft_score, total_hard_score = 0, 0, 0
    # for sample in chosen_caption_dataset:
    #     sample_count += 1
    #     total_soft_score += sample["soft_score"]
    #     total_hard_score += sample["hard_score"]

    # print(f"soft score: {total_soft_score/sample_count}   hard score: {total_hard_score/sample_count}")

    with open(args.output_fn, "w") as f:
        json.dump(chosen_caption_dataset, f, indent=4)

if __name__ == '__main__':
    main()
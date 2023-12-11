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
    if answer == "one world":
        answer = "1 world"
    if answer == "takeoff":
        answer = "take off"
    if answer == "virgin australia":
        answer = "virgin"

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
    question_id_to_context= {}
    for sample in vqa_dataset:
        question_id, image_id = sample['question_id'], sample['image_id']
        context_key = f"{image_id}<->{question_id}"
        question_id_to_context[str(question_id)] = context_key
        answers = get_answers(sample, dataset_type)
        question = sample["question"]
        answer_dict[context_key] = answers
        question_dict[context_key] = question
    return answer_dict, question_dict, question_id_to_context


def load_captions(caption_fn, question_to_context, key="pred_captions"):
    with open(caption_fn) as f:
        caption_dataset = json.load(f)
    caption_dict = {}
    for question_id, sample in caption_dataset.items():
        context_key = question_to_context[question_id]
        captions = sample[key]
        caption_dict[context_key] = captions
    return caption_dict


batchify = lambda x, n: [x[i:i+n] for i in range(0, len(x), n)]


class VQADirectAnswer:
    def __init__(self, args):
        self.args = args

        # load CER evaluator
        self.evaluator = CharErrorRate()

        # loading input questions (and answer for reference accuracy computing)
        self.answer_dict, self.question_dict, val_id_to_context = load_anno(args.val_dataset, args.dataset_type)
        self.val_keys = list(self.question_dict.keys())

        self.train_answer_dict,self.train_question_dict, train_id_to_context = load_anno(args.train_dataset, args.dataset_type)
        self.train_keys = list(self.train_answer_dict.keys())

        # load captions
        self.train_captions = load_captions(args.train_caption_file, train_id_to_context)
        self.val_captions = load_captions(args.val_caption_file, val_id_to_context)
        
        # load tags
        self.tags_dict = self.load_tags()
        
        self.load_similarity()


    def inference(self):
        codex_prompts = []
        codex_answers = []
        
        print("Generating the prompts")
        for key in tqdm(self.val_keys):
            codex_prompts.append(self.sample_prompt(key))

        prompt_batches = batchify(codex_prompts, 19)

        print("Generating the answers by Codex")
        for prompts in tqdm(prompt_batches):
            response = codex(prompts)
            gpt3_answers = [process_answer(sample["text"]) for sample in response["choices"]]
            codex_answers.extend(gpt3_answers)

        inference_log = []

        print("Scoring the answers")
        for key_index, key in tqdm(enumerate(self.val_keys)):
            question, answers = self.question_dict[key], self.answer_dict[key]
            val_caption = self.val_captions[key][0]
            answer = answer_majority(answers)
            image_id = int(key.split('<->')[0])
            question_id = key.split('<->')[1]

            gpt3_answer = process_answer(codex_answers[key_index])

            soft_score, hard_score = self.sample_evaluate(gold_answers=answers, gpt3_answer=gpt3_answer)

            this_sample = { "question_id": question_id,
                            "image_id": image_id,
                            "question": question,
                            "gold_answer": answer,
                            "gpt3_answer": gpt3_answer,
                            "pred_caption": val_caption,
                            "soft_score": soft_score,
                            "hard_score": hard_score,
                            "prompt": codex_prompts[key_index]
                            }
            inference_log.append(this_sample)

        return inference_log


    # get the codex prompt for a given question
    def sample_prompt(self, key):
        question, answers = self.question_dict[key], self.answer_dict[key]
        
        val_caption = self.val_captions[key][0]
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
            
            this_caption = self.train_captions[context_key][0]
            this_tag = self.tags_dict[int(context_key.split('<->')[0])]
            this_question = self.train_question_dict[context_key]
            this_answer = answer_majority(self.train_answer_dict[context_key])

            prompt += f"Context: {this_caption} {this_tag}\n===\n"
            prompt += f"Q: {this_question}\nA: {this_answer}\n\n===\n"
        

        prompt += f"Context: {val_caption} {val_tag}\n===\n"
        prompt += f"Q: {question}\nA:"

        return prompt


    def sample_evaluate(self, gold_answers, gpt3_answer):

        this_answer_scores = []
        count = 0
        for gold_answer in gold_answers:
            this_score = 1 - self.evaluator(gpt3_answer, gold_answer).item()
            this_answer_scores.append(this_score)

            if gpt3_answer == gold_answer:
                count += 1
        soft_score = min(1., sum(sorted(this_answer_scores)[-3:])/3)
        hard_score = min(1., count*0.3)
        return soft_score, hard_score


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
    parser.add_argument('--dataset_type', type=str, default='okvqa', choices=['okvqa', 'vqa2','aokvqa'])
    parser.add_argument('--partition', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--caption_type', type=str, default='caption_tag', help='caption_tag, caption')
    parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--train_caption_file', type=str, default='../data/predicted_captions/okvqa_train_1003.json')
    parser.add_argument('--val_caption_file', type=str, default='../data/predicted_captions/okvqa_val_1003.json')
    parser.add_argument('--train_dataset', type=str, default='../data/task_data/okvqa_w_coco_caption.json')
    parser.add_argument('--val_dataset', type=str, default='../data/task_data/okvqa_w_coco_caption_val.json')
    parser.add_argument('--tag_path', type=str, default='../coco_caption_pred_tags')
    parser.add_argument('--similarity_path', type=str, default='../coco_clip')
    parser.add_argument('--output_fn', default="okvqa_1003_log.json")

    args = parser.parse_args()

    openai.api_key = args.apikey

    pica = VQADirectAnswer(args)

    running_log = pica.inference()

    sample_count, total_soft_score, total_hard_score = 0, 0, 0
    for sample in running_log:
        sample_count += 1
        total_soft_score += sample["soft_score"]
        total_hard_score += sample["hard_score"]

    print(f"soft score: {total_soft_score/sample_count}   hard score: {total_hard_score/sample_count}")

    with open(args.output_fn, "w") as f:
        json.dump(running_log, f, indent=4)

if __name__ == '__main__':
    main()
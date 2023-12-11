import openai
import json
import time
import sys
from tqdm import tqdm
from collections import defaultdict
import argparse


# OpenAI completion
engine = "code-davinci-002"

api_keys = []
class KeyGen:

    def __init__(self) -> None:
        self.key_ind = 0

    def get_key(self):
        self.key_ind += 1
        if self.key_ind >= len(api_keys):
            self.key_ind = 0
        return api_keys[self.key_ind]


key_generator = KeyGen()

# add wrapper for error like throughput limitation
# may try changing to text-davinci-002
# can take a list of prompts as input
def codex(prompt, top_p=1, temperature=0. , n=1):
    response = None
    received = False
    while not received:
        try:
            openai.api_key = key_generator.get_key()
            response = openai.Completion.create(
                engine=engine, 
                prompt=prompt,
                max_tokens=128,
                logprobs=1,
                top_p=top_p,
                n=n,
                temperature=temperature,
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


# return the majority answer from a list
def answer_majority(answers):
    vote_dict = defaultdict(int)
    for a in answers:
        vote_dict[a] += 1
    sorted_answers = [k for k, v in sorted(
        vote_dict.items(), key=lambda item: item[1])]
    return sorted_answers[-1]


# different dataset have different answer format
def get_one_answer(instance, dataset_type):
    answer = ""
    if dataset_type == "vqa2":
        answer = instance['multiple_choice_answer']
    elif dataset_type == "okvqa":
        answer = answer_majority([s['answer'] for s in instance['answers']])
    elif dataset_type == "aokvqa":
        answer = answer_majority(instance['direct_answers'])
    else:
        raise ValueError("dataset not supported")
    return answer


batchify= lambda test_list, batchsize: [test_list[i:i+batchsize] for i in range(0, len(test_list), batchsize)]

in_context_examples = """Summarize the context to help answer the question

Original contexts: A very clean and well decorated empty bathroom. A blue and white bathroom with butterfly themed wall tiles. A bathroom with a border of butterflies and blue paint on the walls above it.
Question: Is the sink full of water?
Answer: no
Summary: A bathroom with an empty sink.

Original contexts: Several metal balls sit in the sand near a group of people.. People standing around many silver round balls on the ground.. Silver balls are lined up in the sand as people mill about in the background.. Silver balls on sand with people walking around. . silver balls laying on the ground around a smaller red ball.
Question: What color are the round objects?
Answer: silver
Summary: People standing around many silver round balls on the ground.

Original contexts: A kitchen cart stands in the middle of a kitchen with wooden cupboards.. A bright kitchen with hardwood floors and wooden cupboards.. a kitchen that is fully furnished with a cart in the center of it.. The kitchen is brightly lit from the window.. Brightly colored kitchen with wooden floors, large images on wall, and small island.
Question: What is the source of light in this picture?
Answer: sun
Summary: A bright kitchen lit by sunlight.

Original contexts: An empty kitchen with white and black appliances.. A refrigerator and stove are in a small kitchen area. . Small kitchen in a personal home with dual sinks.. A small kitchen with sink, stove and refrigerator.. A small kitchen with several appliances and cookware.
Question: How many cabinets in this room?
Answer: 4
Summary: A small kitchen with 4 cabinets.

Original contexts: Green tiled backsplash highlighted by low overhead lighting.. A kitchen counter is illuminated by a hood light. A kitchen sink next to an empty counter with a tiled wall.. A back splash is added to the wall in the kitchen.. A picture of a sink top with dim lighting.
Question: What material is the backsplash made of?
Answer: tile
Summary: Green tiled backsplash highlighted by low overhead lighting.

Original contexts: A graffiti-ed stop sign across the street from a red car . A vandalized stop sign and a red beetle on the road. A red stop sign with a Bush bumper sticker under the word stop.. A stop sign that has been vandalized is pictured in front of a parked car.. A street sign modified to read stop bush.
Question: What season is it in this photo?
Answer: summer
Summary: A stop sign and a car on a street in summer.

Original contexts: Lady carrying a purse walking along side a man.. A city sidewalk is lined with lamp posts. A man and a woman stand on the sidewalk lined with street lights.. A city sidewalk with storefronts on the right.. Two people leaving a building to walk down the street.
Question: Which item in this picture helps people see after dark?
Answer: streetlight
Summary: A city sidewalk lit by streetlight.

Original contexts: A sink and a toilet inside a small bathroom.. White pedestal sink and toilet located in a poorly lit bathroom.. Clean indoor bathroom with tiled floor and good lighting.. a bathroom with toilet and sink and blue wall. a blue bathroom with a sink and toilet.
Question: How many rolls of toilet paper are on the shelves above the toilet?
Answer: 0
Summary: A bathroom with a toilet and a sink. There is no toile paper on the shelves.

Original contexts: A couple enjoying beverages and a snack on a sunny day. Showing a  doughnut while holding drinks near a car.. A man and woman sharing apple cider and a doughnut. Two people are standing in front of an open car trunk holding drinks and a doughnut. . A man and a woman eating donuts and having drinks.. A man holding beer and a woman holding a pastry and beer.
Question: How do we know this guy is not likely to have packed a razor?
Answer: has beard
Summary: A man with beard and a woman are eating donuts and having drinks.

Original contexts: Woman riding a bicycle down an empty street.. A woman in green is riding a bike.. a woman wearing a bright green sweater riding a bicycle. A woman on a bicycle is going down the small town street.. A woman bikes down a one way street.
Question: What kind of fruit is the helmet supposed to be?
Answer: watermelon
Summary: A woman with a watermelon style helmet riding a bicycle.

Original contexts: A panoramic view of a kitchen and all of its appliances. A panoramic photo of a kitchen and dining room A wide angle view of the kitchen work area multiple photos of a brown and white kitchen.  A kitchen that has a checkered patterned floor and white cabinets.
Question: Is the counter curved?
Answer: no
Summary: A photo of a kitchen with a counter that is not curved.

Original contexts: A woman is walking a dog in the city.. A woman and her dog walking down a sidewalk next to a fence with some flowers. . A woman walking her dog on the sidewalk.. A woman walks her dog along a city street.. A woman walks her dog on a city sidewalk.
Question: What color vehicle is closest to the mailbox?
Answer: silver
Summary: A silver vehicle next to a mailbox on the sidewalk.

Original contexts: some pancakes cover with bananas, nuts, and some whipped cream . Two pancakes on top of a white plate covered in whipped cream, nuts and a banana.. Pancakes with bananas, nuts and cream, covered in syrup. . Pancakes topped with bananas, whipped cream and walnuts.. Pancakes topped with bananas, nuts, and ice cream.
Question: What restaurant was this dish cooked at?
Answer: ihop
Summary: Pancakes with banans, nuts, and cream, cooked at ihop.

Original contexts: The two people are walking down the beach.. Two people carrying surf boards on a beach.. Two teenagers at a white sanded beach with surfboards.. A couple at the beach walking with their surf boards.. A guy and a girl are walking on the beach holding surfboards.
Question: What is on the man's head?
Answer: hat
Summary: A man and a woman walking on the beach with surfboards. The man is wearing a hat.

Original contexts: A sink and a toilet inside a small bathroom.. White pedestal sink and toilet located in a poorly lit bathroom.. Clean indoor bathroom with tiled floor and good lighting.. a bathroom with toilet and sink and blue wall. a blue bathroom with a sink and toilet.
Question: Is there natural light in this photo?
Answer: no
Summary: A photo of a small bathroom in artificial light.

Original contexts: Fog is in the air at an intersection with several traffic lights.. An intersection during a cold and foggy night.. Empty fog covered streets in the night amongst traffic lights.. City street at night with several stop lights.. It is a foggy night by a traffic light.
Question: Which direction is okay to go?
Answer: straight
Summary: A traffic light in a foggy night, showing it is okay to go straight.

Original contexts: A graffiti-ed stop sign across the street from a red car . A vandalized stop sign and a red beetle on the road. A red stop sign with a Bush bumper sticker under the word stop.. A stop sign that has been vandalized is pictured in front of a parked car.. A street sign modified to read stop bush.
Question: What color is the car driving north?
Answer: red
Summary: A stop sign and a red car driving north.

Original contexts: A man in a wheelchair and another sitting on a bench that is overlooking the water.. Two people sitting on dock looking at the ocean.. Two older people sitting down in front of a beach.. An old couple at the beach during the day.. A person on a bench, and one on a wheelchair sitting by a seawall looking out toward the ocean.
Question: What is the person on the left sitting on?
Answer: bench
Summary: A person sit on a bench on the left, and another sitting in a wheelchair on the right, all looking at the ocean.

Original contexts: A parked motor scooter sitting next to a bicycle.. A picture of a motorbike and two pedal bicycles.. A motor scooter that has an advertisment on the back next to a bicycle.. A grey moped parked by building next to a bicycle.. a motor bike parked next to a bike by a building.
Question: Which model of bike is shown in this picture?
Answer: vespa
Summary: A motor scotter parking next to a vespa bike.

Original contexts: People standing around a park bench next to a bicycle.. A group of women are describing a new setup for a building plan. a group of people in a field of grass near a building. Several people standing in an area with picnic tables looking at a board.. A woman giving a presentation in a park.
Question: What is the woman in the blue jacket standing on?
Answer: bench
Summary: A woman in blue jacket standing on a bench, with a group of people around her.

Original contexts: A orange tabby cat laying down on a black car. An orange cat laying on the hood on a car.. A cat sits on top of a black car.. A cat that is sitting on top of a black car.. A yellow cat sleeping on the hood of a black car parked in the garage.
Question: What brand of car is this?
Answer: subaru
Summary: An orange cat laying on top of a black subaru.

Original contexts: A bicycle parked in front of a building next to a pile of garbage.. Black and white photograph of a homeless person under their many belongings. Two people huddle on a bench under their belongings.. A homeless person is bundled within a pile of belongings..  an image of two homeless people laying under debris on a bench
Question: How is the bike affixed to the pole?
Answer: chain
Summary: A bicycle parked in from of a building, attaching the pole with a chain.

"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True, help="input dataset file")
    parser.add_argument("--dataset", type=str, required=True, choices="okvqa, vqa2, aokvqa", help="type of dataset")
    parser.add_argument("--output", type=str, required=True, help="output file name")
    # for diverse generation, use n=10, t=1.0, top_p=0.96
    args = parser.parse_args()

    # read the input file
    with open(args.input) as f:
        vqa_dataset = json.load(f)
    
    if args.dataset != "aokvqa":
        vqa_dataset = vqa_dataset["annotations"]

    prompts = []

    for i in range(len(vqa_dataset)):
        prompt = in_context_examples
        prompt += f"Original contexts: {'. '.join([it['caption'] for it in vqa_dataset[i]['caption']])}\n"
        prompt += f"Question: {vqa_dataset[i]['question']}\n"

        answer = get_one_answer(vqa_dataset[i], dataset_type=args.dataset)

        prompt += f"Answer: {answer}\n"
        prompt += "Summary:"

        prompts.append(prompt)

    batched_prompts = batchify(prompts, 19)


    answers = []
    # codex completion
    for batch in tqdm(batched_prompts):
        resp = codex(batch, temperature=0.0, n=1)
        batch_answer = [sample["text"] for sample in resp["choices"]]
        # batch_answer = batchify(batch_answer, 19)
        answers += batch_answer

        # backup answers
        with open("answer_saving.tmp", "w") as f:
            json.dump(answers, f, indent=4)

    # save the answers
    new_vqa_dataset = []

    for i in range(len(vqa_dataset)):
        this_item = {}
        this_item["question_id"] = vqa_dataset[i]["question_id"]
        this_item["image_id"] = vqa_dataset[i]["image_id"]
        this_item["question"] = vqa_dataset[i]["question"]
        this_item["answer"] = get_one_answer(vqa_dataset[i], args.dataset)
        this_item['coco_captions'] = [it['caption'] for it in vqa_dataset[i]['caption']]
        # remove leading space
        this_item['prompt_guided_captions'] = answers[i]
        new_vqa_dataset.append(this_item)

    with open(args.output, "w") as f:
        json.dump(new_vqa_dataset, f, indent=4)

if __name__ == "__main__":
    main()

    



    

    

    
    

# PromptCap
This repository contains the code and models for our paper [PromptCap: Prompt-Guided Task-Aware Image Captioning](https://arxiv.org/abs/2211.09699). Please refer to the [project page](https://yushi-hu.github.io/promptcap_demo/) for a quick overview. This paper is also accepted to ICCV 2023, with title [PromptCap: Prompt-Guided Image Captioning for VQA with GPT-3](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_PromptCap_Prompt-Guided_Image_Captioning_for_VQA_with_GPT-3_ICCV_2023_paper.html).

# Replicating results
Since Codex has been deprecated, it is hard to replicate the results for PromptCap. For ease of use, we release all our logs, with the prompts we give to GPT-3 (codex), and the GPT-3's answers for each question, in `Evaluation Logs`.

# QuickStart

## Installation
```
pip install promptcap
```

Two pipelines are included. One is for image captioning, and the other is for visual question answering.

## Captioning Pipeline

Please follow the prompt format, which will give the best performance.

Generate a prompt-guided caption by following:
```python
import torch
from promptcap import PromptCap

model = PromptCap("vqascore/promptcap-coco-vqa")  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"

if torch.cuda.is_available():
  model.cuda()

prompt = "please describe this image according to the given question: what piece of clothing is this boy putting on?"
image = "glove_boy.jpeg"

print(model.caption(prompt, image))
```

To try generic captioning, just use "what does the image describe?"

```python
prompt = "what does the image describe?"
image = "glove_boy.jpeg"

print(model.caption(prompt, image))
```



PromptCap also support taking OCR inputs:

```python
prompt = "please describe this image according to the given question: what year was this taken?"
image = "dvds.jpg"
ocr = "yip AE Mht juor 02/14/2012"

print(model.caption(prompt, image, ocr))
```



## Visual Question Answering Pipeline

Notice: This is not the pipeline we used for the paper, please reference to the `Replicating Results` section to get our GPT-3 result.

Different from typical VQA models, which are doing classification on VQAv2, PromptCap is open-domain and can be paired with arbitrary text-QA models.
Here we provide a pipeline for combining PromptCap with UnifiedQA.

```python
import torch
from promptcap import PromptCap_VQA

# QA model support all UnifiedQA variants. e.g. "allenai/unifiedqa-v2-t5-large-1251000"
vqa_model = PromptCap_VQA(promptcap_model="vqascore/promptcap-coco-vqa", qa_model="allenai/unifiedqa-t5-base")

if torch.cuda.is_available():
  vqa_model.cuda()

question = "what piece of clothing is this boy putting on?"
image = "glove_boy.jpeg"

print(vqa_model.vqa(question, image))
```

Similarly, PromptCap supports OCR inputs

```python
question = "what year was this taken?"
image = "dvds.jpg"
ocr = "yip AE Mht juor 02/14/2012"

print(vqa_model.vqa(question, image, ocr=ocr))
```

Because of the flexibility of Unifiedqa, PromptCap also supports multiple-choice VQA

```python
question = "what piece of clothing is this boy putting on?"
image = "glove_boy.jpeg"
choices = ["gloves", "socks", "shoes", "coats"]
print(vqa_model.vqa_multiple_choice(question, image, choices))
```

# Reference codes for re-training PromptCap.
We provide the original codes we use for PromptCap in `Original Codes`.
Notice that this is not a runnable pipeline because codex and OpenAI text completion are deprecated, and the CLIP embeddings for the whole coco are too big.
Nevertheless, it is still valuable for follow-up works to know the details of our implementation.

1. Training data generation: refer to `Original Codes/promptcap-gen` on how we generate the prompt-guided caption training data with Codex.
2. Training data filtering: refer to `Original Codes/example-filtering` on how we filter the training data.
3. Training PromptCap with GPT-3 synthesized data:

We release the training data synthesized by Codex in `vqa2_train_1010.zip`. To train PromptCap from OFA, first process the data according to [add a task](https://github.com/OFA-Sys/OFASys/blob/main/docs/source/howto/add_task.rst) and then fine-tune according to [how to train](https://github.com/OFA-Sys/OFASys/blob/main/docs/source/howto/train.rst). As the field is developing so quickly, we recommend train PromptCap with newer vision-language models, like BLIP-2 and LLaVA.

4. Inference on GPT-3: the prompt logs are in `Evaluation Logs`. For construction of the prompts, refer to `Original Codes/GPT3-inference`.

## Bibtex
```
@article{hu2022promptcap,
  title={PromptCap: Prompt-Guided Task-Aware Image Captioning},
  author={Hu, Yushi and Hua, Hang and Yang, Zhengyuan and Shi, Weijia and Smith, Noah A and Luo, Jiebo},
  journal={arXiv preprint arXiv:2211.09699},
  year={2022}
}
```

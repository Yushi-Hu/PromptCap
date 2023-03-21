# PromptCap
Prompt-Guided Image Captioning


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

## Bibtex
```
@article{hu2022promptcap,
      title={PromptCap: Prompt-Guided Task-Aware Image Captioning},
      author={Hu, Yushi* and Hua, Hang* and Yang, Zhengyuan and Shi, Weijia and Smith, Noah A and Luo, Jiebo},
      journal={arXiv preprint arXiv:2211.09699},
      year={2022}
```

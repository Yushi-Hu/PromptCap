import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from .modeling_ofa import OFAModel
from .tokenization_ofa import OFATokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

class PromptCap(nn.Module):
    def __init__(self, ckpt="vqascore/promptcap-coco-vqa"):
        super().__init__()
        self.tokenizer = OFATokenizer.from_pretrained(ckpt)
        self.model = OFAModel.from_pretrained(ckpt, use_cache=True)
        self.model.eval()
        
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 480
        
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution),
                              transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def caption(self, prompt, image, num_beams=5, no_repeat_ngram_size=3, max_new_tokens=100, **generator_args):
        image = Image.open(image)
        image = self.patch_resize_transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.model.device)
        
        prompt = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt = prompt.to(self.model.device)
        
        with torch.no_grad():
            gen = self.model.generate(prompt, patch_images=image, 
                                      num_beams=num_beams, 
                                      no_repeat_ngram_size=no_repeat_ngram_size, 
                                      max_new_tokens=max_new_tokens,
                                      **generator_args)
            
        return (self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]).strip()

class PromptCap_VQA(nn.Module):
    def __init__(self, promptcap_model="vqascore/promptcap-coco-vqa",
                 qa_model="allenai/unifiedqa-v2-t5-large-1251000"):
        super().__init__()
        
        self.captioner = PromptCap(promptcap_model)
        
        # QA model
        self.tokenizer = T5Tokenizer.from_pretrained(qa_model)
        self.model = T5ForConditionalGeneration.from_pretrained(qa_model)
        self.model.eval()
        
    def run_model(self, input_string, max_new_tokens=50, **generator_args):
        with torch.no_grad():
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
            res = self.model.generate(input_ids.to(self.model.device), max_new_tokens=max_new_tokens, **generator_args)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)
        
    def vqa(self, question, image, ocr="", **generator_args):
        prompt = f"please describe this image according to the given question: {question}"
        
        if ocr:
            prompt += f" OCR inputs: {ocr}"
        
        caption = self.captioner.caption(prompt, image)
        answer = self.run_model(
            f"{question} \n {caption}", **generator_args)[0]
        
        return answer.strip()
    
    def vqa_multiple_choice(self, question, image, choices=[], **generator_args):
        
        prompt = f"please describe this image according to the given question: {question}"
        caption = self.captioner.caption(prompt, image)
        
        if len(choices) > 0:
            choice_text = ""
            for i in range(len(choices)):
                choice_text += f"({i+1}) {choices[i]} "

        return self.run_model(f"{question} \n {caption} \n {choice_text}", **generator_args)[0]
        

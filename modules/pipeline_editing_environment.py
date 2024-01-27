from typing import List

import PIL
import requests

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


class EditingEnvironmentPipeline:
    """
    Pipeline for editing images' global attributes based on InstructPix2Pix.
    """
    def __init__(self, name='timbrooks/instruct-pix2pix', device='cuda'):
        super().__init__()
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(name, torch_dtype=torch.float32, safety_checker=None)
        self.model.to(device)
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(self.model.scheduler.config)
    
    def __call__(self, image: List[PIL.Image.Image], text: List[str], steps=50, **kwargs):
        return self.model(text, image=image, num_inference_steps=steps, **kwargs).images
    

if __name__ == '__main__':
    image = PIL.Image.open('/home/gtangg12/auto-augment/tests/example1.png')
    text1 = 'turn the road conditions to snowy'
    text2 = 'turn the road conditions to rainy'
    text3 = 'turn the road conditions to nighttime'
    pipeline = EditingEnvironmentPipeline()
    output = pipeline([image, image, image], [text1, text2, text3])
    for i, image_out in enumerate(output):
        image_out.save(f'/home/gtangg12/auto-augment/tests/example{i}_output.png')
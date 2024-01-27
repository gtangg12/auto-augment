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
    
    def __call__(
        self, 
        text:  List[str], 
        image: List[PIL.Image.Image], steps=50, **kwargs
    ):
        """
        :param steps: number of diffusion steps to sample
        """
        with torch.no_grad():
            outputs = self.model(text, image=image, num_inference_steps=steps, **kwargs)
        return outputs.images
    

if __name__ == '__main__':
    image = PIL.Image.open('/home/gtangg12/auto-augment/tests/example.png')
    text1 = 'turn the road conditions to snowy'
    text2 = 'turn the road conditions to rainy'
    text3 = 'turn the road conditions to nighttime'
    pipeline = EditingEnvironmentPipeline()
    outputs = pipeline(
        [text1, text2, text3], 
        [image, image, image],
    )
    for i, image_out in enumerate(outputs):
        image_out.save(f'/home/gtangg12/auto-augment/tests/example{i}_output.png')
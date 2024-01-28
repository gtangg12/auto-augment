from typing import List

import PIL
import requests

import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)


class EditingEnvironmentPipeline:
    """
    Pipeline for editing images' global attributes based on InstructPix2Pix.
    """

    def __init__(self, name="timbrooks/instruct-pix2pix", device="cuda"):
        super().__init__()
        self.model = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            name, torch_dtype=torch.float32, safety_checker=None
        )
        self.model.to(device)
        self.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.model.scheduler.config
        )

    def __call__(
        self,
        text: List[str],
        image: List[PIL.Image.Image],  # w, h
        steps=50,
        **kwargs,
    ) -> List[PIL.Image.Image]:  # w, h
        # batched inference, same number of text, image and outputs
        """
        :param steps: number of diffusion steps to sample
        """
        if isinstance(text, str):
            text = [text]
        if isinstance(image, PIL.Image.Image):
            image = [image]

        with torch.no_grad():
            outputs = self.model(text, image=image, num_inference_steps=steps, **kwargs)
        return [x.resize(image[0].size) for x in outputs.images]


if __name__ == "__main__":
    image = PIL.Image.open("/home/gtangg12/auto-augment/tests/example.png")
    text1 = "turn the road conditions to snowy"
    text2 = "turn the road conditions to rainy"
    text3 = "turn the road conditions to nighttime"
    pipeline = EditingEnvironmentPipeline()
    outputs = pipeline(
        [text1, text2, text3],
        [image, image, image],
    )
    for i, image_out in enumerate(outputs):
        image_out.save(f"/home/gtangg12/auto-augment/tests/example_output{i}.png")
    print("original image shape", image.size)
    print("output image shape", outputs[0].size)

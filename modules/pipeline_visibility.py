from typing import List

import PIL
import numpy as np
import torch
import torch.nn as nn
from transformers import DPTImageProcessor, DPTForDepthEstimation


class DepthModel:
    """
    Model for computing depth maps from images based on DPT (Dense Prediction Transformer).
    """
    def __init__(self, name='Intel/dpt-large', device='cuda'):
        self.processor = DPTImageProcessor.from_pretrained(name)
        self.model = DPTForDepthEstimation.from_pretrained(name)
        self.model.to(device)
        self.model.eval()

    def __call__(self, image: List[PIL.Image.Image]):
        """
        Computes and interpolates depth maps back to the original image size.
        """
        if isinstance(image, PIL.Image.Image):
            image = [image]
        
        inputs = self.processor(images=image, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        prediction = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1), size=image[0].size[::-1], mode='bicubic', align_corners=False
        )
        return [pred.squeeze().cpu().numpy() for pred in prediction]
    
    @classmethod
    def calibrate(cls, pred: np.ndarray, scale=100, shift=-7):
        return scale / (pred + shift)
    
    @classmethod
    def colormap(cls, pred: np.ndarray):
        pred = pred / np.max(pred) * 255
        pred = PIL.Image.fromarray(pred.astype('uint8'))
        pred = pred.convert('L')
        return pred
    

if __name__ == '__main__':
    image = PIL.Image.open('/home/gtangg12/auto-augment/tests/example.png')
    model = DepthModel()
    outputs = model(image)
    for i, depth in enumerate(outputs):
        depth_out = DepthModel.calibrate(depth)
        np.save(f'/home/gtangg12/auto-augment/tests/example{i}_output_depth.npy', depth_out)
        image_out = DepthModel.colormap(depth)
        image_out.save(f'/home/gtangg12/auto-augment/tests/example{i}_output_depth.png')
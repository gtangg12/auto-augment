from typing import List

import PIL
import numpy as np
import torch
import torch.nn as nn
from transformers import DPTImageProcessor, DPTForDepthEstimation
from modules.pipeline_visibility_utils import *


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
        prediction = torch.clip(prediction, 0, 1000)
        return [pred.squeeze().cpu().numpy() for pred in prediction]
    
    @classmethod
    def calibrate(cls, pred: np.ndarray, scale=100, shift=0):
        depth = scale / (pred + shift + 1e-5)
        depth = np.clip(depth, 0, 1000)
        return depth
    
    @classmethod
    def colormap(cls, pred: np.ndarray):
        pred = pred / np.max(pred) * 255
        pred = PIL.Image.fromarray(pred.astype('uint8'))
        pred = pred.convert('L')
        return pred
    

class VisibilityDegredationModel:
    """
    Model for simulating visibility degredation.
    """
    def __init___(self):
        self.model_depth = DepthModel()

    def __call__(self, image: List[PIL.Image.Image], depth: List[np.ndarray]=None, **kwargs):
        """
        :param image: list of PIL.Image.Image
        :param depth: list of np.ndarray of depth (optionally)
        **kwargs for gaussian_source_sink (see pipeline_visibility_utils.py)
        """
        depth = depth or [DepthModel.calibrate(x) for x in self.model_depth(image)]
        outputs = []
        for x, xd in zip(image, depth):
            xi = np.array(x) / 255
            xi = xi.transpose(2, 0, 1)
            betas = gaussian_source_sink(xi.shape[1:], **kwargs)
            xo = haze(xi, xd, beta=betas)
            xo = xo.transpose(1, 2, 0)
            xo = PIL.Image.fromarray((xo * 255).astype('uint8'))
            outputs.append(xo)
        return outputs
    

if __name__ == '__main__':
    image = PIL.Image.open('/home/gtangg12/auto-augment/tests/example.png')
    model = DepthModel()
    outputs = model(image)
    depth = outputs[0]
    depth_out = DepthModel.calibrate(depth)
    np.save(f'/home/gtangg12/auto-augment/tests/example_output_depth.npy', depth_out)
    image_out = DepthModel.colormap(depth)
    image_out.save(f'/home/gtangg12/auto-augment/tests/example_output_depth.png')

    model_degrader = VisibilityDegredationModel() # aka model_pua
    image = [image]
    depth = [depth_out]
    outputs = model_degrader(image, depth, beta=0.1, num_gaussians=64, source_sink_ratio=0.5, max_scale=0.025, mode='smooth')
    outputs[0].save(f'/home/gtangg12/auto-augment/tests/example_output_haze.png')
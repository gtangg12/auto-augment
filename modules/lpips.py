from typing import List
import torch
from PIL import Image
import torchvision
import lpips


class LPIPS:
    def __init__(self):
        self.lpips = lpips.LPIPS(net="vgg")

    def _convert_image(
        self,
        image: Image.Image,  # w, h
    ) -> torch.Tensor:  # 3, h, w, [-1, 1]
        return (torchvision.transforms.functional.to_tensor(image) * 2 / 255) - 1

    def lpips_similarity(
        self,
        images1: List[Image.Image],  # w, h
        images2: List[Image.Image],  # w, h
    ) -> List[float]:
        batch1 = torch.stack([self._convert_image(image) for image in images1], dim=0)
        batch2 = torch.stack([self._convert_image(image) for image in images2], dim=0)
        res = self.lpips(batch1, batch2).squeeze((1, 2, 3)).tolist()
        if len(images1) == 1:
            return [res]
        return res

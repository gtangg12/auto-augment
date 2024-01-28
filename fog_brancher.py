from typing import List
from modules.pipeline_visibility import VisibilityDegredationModel
from PIL import Image
import random


class FogBrancher:
    def __init__(self):
        # lol i love to wRaP mY cLaSsEs
        self.delegate = VisibilityDegredationModel()
    
    def __call__(self, image: Image.Image):# -> List[Image.Image]:
        res = []
        for _ in range(4):
            beta = random.random() * 0.5
            num_gaussians = random.randint(32, 128)
            source_sink_ratio = random.random()
            res.extend(self.delegate([image], beta=beta, num_gaussians=num_gaussians, source_sink_ratio=source_sink_ratio, max_scale=0.025, mode='smooth'))
        yield res

if __name__ == '__main__':
    image = Image.open('tests/example.png')
    brancher = FogBrancher()
    for x in brancher(image):
        for i, y in enumerate(x):
            y.save(f'tests/example_output_{i}.png')
import PIL.Image
from typing import List, Tuple

class FakeBranchingAgent:
    def branch(self, image: PIL.Image.Image): # see brancher -> Tuple[List[str], List[PIL.Image.Image]]:
        yield ['no tactic', 'no tactic']
        yield [image, image]
    

class FakeFogBranchingAgent:
    def branch(self, image: PIL.Image.Image): # see brancher -> Tuple[List[str], List[PIL.Image.Image]]:
        yield [image]
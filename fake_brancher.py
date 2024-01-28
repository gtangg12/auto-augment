import PIL.Image
from typing import List
class FakeBranchingAgent:
    def branch(self, image: PIL.Image.Image) -> List[PIL.Image.Image]:
        return [image]
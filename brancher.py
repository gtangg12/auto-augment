import PIL
from typing import List
from modules.lpips import LPIPS
from modules.pipeline_editing_environment import EditingEnvironmentPipeline

from tactic_service import TacticService


class BranchingAgent:
    """ """

    def __init__(self, score_threshold: float = 1e-2):
        self.tactic_service = TacticService()
        self.pix2pix_service = EditingEnvironmentPipeline()
        self.lpips_service = LPIPS()
        self.score_threshold = score_threshold

    def branch(self, image: PIL.Image.Image) -> List[PIL.Image.Image]:
        """ """
        tactics = self.tactic_service(image)
        if len(tactics) == 0:
            print('no tactics found')
            return [] # no image generated
        augments = self.pix2pix_service(text=tactics, image=[image for _ in tactics])
        scores = self.lpips_service([image for _ in tactics], augments)
        results = [
            augment
            for augment, score in zip(augments, scores)
            if score < self.score_threshold
        ]
        if len(results) == 0:
            print('no pix2pix image passed validation')
            return []
        return results

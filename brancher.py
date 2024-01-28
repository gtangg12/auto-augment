import PIL
from typing import List, Tuple
from modules.lpips import LPIPS
from modules.pipeline_editing_environment import EditingEnvironmentPipeline

from tactic_service import TacticService



class BranchingAgent:
    """ """

    def __init__(self, score_threshold: float = 1e-2, n_tactics: int = 4):
        self.tactic_service = TacticService(n_tactics=n_tactics)
        self.pix2pix_service = EditingEnvironmentPipeline()
        self.lpips_service = LPIPS()
        self.score_threshold = score_threshold

    def branch(self, image: PIL.Image.Image): # yields a tuple of string, then yields images, then returns -> Tuple[List[str], List[PIL.Image.Image]]:
        """ """
        tactics = self.tactic_service(image)
        if len(tactics) == 0:
            print('no tactics found')
            yield []
            yield []
            return # no image generated
        yield tactics
            
        augments = self.pix2pix_service(text=tactics, image=[image for _ in tactics])
        scores = self.lpips_service([image for _ in tactics], augments)
        results = [
            augment
            for augment, score in zip(augments, scores)
            if score < self.score_threshold
        ]
        if len(results) == 0:
            print('no pix2pix image passed validation')
        yield results

    def make_lpips_tuneset(self, image: PIL.Image.Image):
        tactics = self.tactic_service(image)
        if len(tactics) == 0:
            print('no tactics found')
            return
            
        augments = self.pix2pix_service(text=tactics, image=[image for _ in tactics])
        scores = self.lpips_service([image for _ in tactics], augments)
        return augments, scores
    

if __name__ == '__main__':
    image = PIL.Image.open('tests/example.png')
    agent = BranchingAgent(n_tactics=8)

    all_scores = []
    for i in range(5):
        augments, scores = agent.branch(image)
        for a, s in zip(augments, scores):
            a.save(f'tests/example_tunelpip_{len(all_scores)}.png')
            all_scores.append(s)

    print(all_scores)
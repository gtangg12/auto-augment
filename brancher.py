import os
import random
import PIL
from typing import List, Tuple
from modules.lpips import LPIPS
from modules.pipeline_editing_environment import EditingEnvironmentPipeline

from tactic_service import TacticService



class BranchingAgent:
    """ """

    def __init__(self, base_image: PIL.Image.Image, score_threshold: float = 0.003, n_tactics: int = 4):
        self.tactic_service = TacticService(n_tactics=n_tactics)
        self.pix2pix_service = EditingEnvironmentPipeline()
        self.lpips_service = LPIPS()
        self.score_threshold = score_threshold
        self.base_image = base_image
        self.save_to = 'generated/'
        os.mkdir(self.save_to)

    def save_data(self, image: PIL.Image.Image, tactic: str, score: float):
        image.save(
            os.path.join(
                self.save_to, 
                f'data_{tactic}_{score}_{random.randint(0, int(1e6))}'.replace(' ', '_').replace('.', '_')
                  + '.png'))

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
        scores = self.lpips_service([self.base_image for _ in tactics], augments)
        results = [
            augment
            for augment, score in zip(augments, scores)
            if score < self.score_threshold
        ]
        filtered = [
            tactic
            for tactic, score in zip(tactics, scores)
            if score > self.score_threshold
        ]
        for augment, tactic, score in zip(augments, tactics, scores):
            self.save_data(augment, tactic, score)
        if len(filtered) > 0:
            new_augments = self.pix2pix_service(text=filtered, image=[image for _ in filtered], guidance_scale=3)
            new_scores = self.lpips_service([self.base_image for _ in filtered], new_augments)
            results.extend([
                augment
                for augment, score in zip(new_augments, new_scores)
                if score < self.score_threshold
            ])
            failed_tactics = [
                tactic
                for tactic, score in zip(filtered, new_scores)
                if score > self.score_threshold
            ]
            for augment, tactic, score in zip(new_augments, filtered, new_scores):
                self.save_data(augment, tactic, score)
            if len(failed_tactics) > 0:
                print('failed tactics:')
                print(failed_tactics)
        if len(results) == 0:
            print('no pix2pix image passed validation')
        yield results

    def make_lpips_tuneset(self, image: PIL.Image.Image):
        tactics = self.tactic_service(image)
        if len(tactics) == 0:
            print('no tactics found')
            return [], []
            
        augments = self.pix2pix_service(text=tactics, image=[image for _ in tactics])
        scores = self.lpips_service([self.base_image for _ in tactics], augments)
        return augments, scores
    

if __name__ == '__main__':
    image = PIL.Image.open('tests/example.png')
    agent = BranchingAgent(image, n_tactics=4)

    all_scores = []
    for i in range(10):
        augments, scores = agent.make_lpips_tuneset(image)
        for a, s in zip(augments, scores):
            a.save(f'tests/example_tunelpip_{len(all_scores)}.png')
            print(s)
            all_scores.append(s)

    print(all_scores)
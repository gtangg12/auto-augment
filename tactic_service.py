"""The tactic service generates tactics for data augmentation."""


from typing import List
from brancher_prompt import TACTIC_GENERATION_PROMPT
from modules.model_gpt import GPT, SystemMode
from PIL import Image


class TacticService:
    def __init__(self, n_tactics: int = 10):
        self.n_tactics = n_tactics

    def __call__(self, image: Image.Image) -> List[str]:
        for _ in range(3):
            model = GPT(
                system_mode=SystemMode.MAIN,
                system_text=TACTIC_GENERATION_PROMPT,
            )
            response = model.forward(image=[image])
            lines = response.split("\n")
            if all(lines[i].startswith(f"{i+1}.") for i in range(self.n_tactics)):
                return [
                    line.split(" ", 1)[1].strip() for line in lines[: self.n_tactics]
                ]
        print("tactic generation failed after 3 attempts")
        return []

"""The tactic service generates tactics for data augmentation."""


import random
from typing import List
from brancher_prompt import TACTIC_GENERATION_PROMPT
from modules.model_gpt import GPT, SystemMode
from PIL import Image


class TacticService:
    def __init__(self, n_tactics: int = 4):
        self.n_tactics = n_tactics

    def __call__(self, image: Image.Image) -> List[str]:
        for _ in range(3):
            try:
                prompt = TACTIC_GENERATION_PROMPT.format(str(self.n_tactics + 6))
                model = GPT(
                    system_mode=SystemMode.MAIN,
                    system_text=prompt,
                )
                print("tactic generation prompt: ", prompt)
                response = model.forward(image=[image])
                lines = response.split("\n")
                if all(lines[i].startswith(f"{i+1}.") for i in range(self.n_tactics)):
                    tactics = [
                        line.split(" ", 1)[1].strip() for line in lines
                    ]
                    tactics = [x for x in tactics if 'traffic' not in x.lower() and "saturation" not in x.lower()]
                    tactics = random.sample(tactics, self.n_tactics)
                    return tactics
            except Exception:
                pass
            print("attempt failed for tactic generation. this is rare: llm output: ", response)
        print("tactic generation failed after 3 attempts")
        return []

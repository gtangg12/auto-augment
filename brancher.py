import PIL
from typing import List

from modules.model_gpt import GPT, SystemMode
from brancher_prompt import TACTIC_GENERATION_PROMPT


class BranchingAgent:
    """
    """
    def __init__(self):
        self.model = GPT(
            system_mode=SystemMode.MAIN,
            system_text=TACTIC_GENERATION_PROMPT,
        )

    def __call__(self, image: PIL.Image.Image):
        """
        """
        data = []
        return data

    def branch(image: PIL.Image.Image) -> List[PIL.Image.Image]:
        """
        """
        pass
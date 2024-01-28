from brancher import BranchingAgent
from PIL import Image

branch = BranchingAgent()
image = Image.open("tests/example.png")
branches = branch.branch(image)
for i, b in enumerate(branches):
    b.save(f"tests/example_brancher_{i}.png")

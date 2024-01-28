from brancher import BranchingAgent
from PIL import Image

branch = BranchingAgent()
image = Image.open("tests/example.png")
branches = branch.branch(image)
idx = 0
for i, b in enumerate(branches):
    if idx == 1:
        b.save(f"tests/example_brancher_{i}.png")
    idx += 1

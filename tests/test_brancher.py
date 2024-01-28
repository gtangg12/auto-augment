from brancher import BranchingAgent
from PIL import Image

branch = BranchingAgent()
image = Image.open("tests/example.png")
branches = branch.branch(image)
for i, b in enumerate(branches):
    if i == 1:
        for j, ima in enumerate(b):
            ima.save(f"tests/example_brancher_{j}.png")

from brancher import BranchingAgent
from PIL import Image

image = Image.open("tests/example.png")
branch = BranchingAgent(image, n_tactics=1)
branches = branch.branch(image)
for i, b in enumerate(branches):
    if i == 1:
        for j, ima in enumerate(b):
            ima.save(f"tests/example_brancher_{j}.png")

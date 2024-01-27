from tactic_service import TacticService
from PIL import Image


if __name__ == "__main__":
    service = TacticService()
    # I wonder which idiot put data in tests/ folder hmmmm.....
    image = Image.open("tests/example.png")
    tactics = service(image)
    print(tactics)

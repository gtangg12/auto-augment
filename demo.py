# from brancher import BranchingAgent
from typing import List
from fake_brancher import FakeBranchingAgent, FakeFogBranchingAgent
import gradio as gr
import os
import time
from PIL import Image

# agent = BranchingAgent()
agent = FakeBranchingAgent()
fogAgent = FakeFogBranchingAgent()
images = {}
tmpdir: str = ""
def add_file(messages, file):
    """
    """
    print(messages)
    images[file.name] = Image.open(file)
    messages.append([[file.name,], None])
    global tmpdir
    tmpdir = os.path.dirname(file.name)
    yield messages

counter = 0
def new_image_path() -> str:
    global counter
    counter += 1
    return os.path.join(tmpdir, f".__wzhao6_internal__{counter}.png")
  
def autoBot(messages):
  # shittiest code ive ever written
  queue = [images[messages[-1][0][0]]]
  print(messages)
  while True:
    image = queue.pop(0)
    messages.append([None, "**Generating Fog Augmentations**"])
    yield messages
    for results in fogAgent.branch(image):
      branches: List[Image.Image] = results
      queue.extend(branches)
      handleImages_(messages, branches)
      yield messages
    messages.append([None, "**Generating Augmentations**"])
    yield messages
    idx = 0
    for results in agent.branch(image):
      if idx == 0:
        tactics: List[str] = results
        print('got tactics', tactics)
        messages.append(
           [None, f"LLM Suggested Augmentations: {', '.join(tactics)}"]
        )
      elif idx == 1:
        branches: List[Image.Image] = results
        queue.extend(branches)
        handleImages_(messages, branches)
      idx += 1
      yield messages

def handleImages_(messages, branches: List[Image.Image]):
    names = []
    for output in branches:
      path = new_image_path()
      output.save(path)
      names.append(path)
    print('got branches', names)
    messages.append(
      [names, None]
    )

def fogBot(messages):
    print(messages)
    messages.append([None, "**Generating Fog Augmentations**"])
    yield messages
    for results in fogAgent.branch(images[messages[-2][0][0]]):
        branches: List[Image.Image] = results
        handleImages_(messages, branches)
        yield messages

def bot(messages):
    """
    """
    print(messages)
    messages.append([None, "**Generating Augmentations**"])
    yield messages
    idx = 0
    for results in agent.branch(images[messages[-2][0][0]]):
      if idx == 0:
        tactics: List[str] = results
        print('got tactics', tactics)
        messages.append(
           [None, f"LLM Suggested Augmentations: {', '.join(tactics)}"]
        )
      elif idx == 1:
        branches: List[Image.Image] = results
        handleImages_(messages, branches)
      idx += 1
      yield messages

CSS = """
.contain {
  display: flex;
  flex-direction: column;
}

.gradio-container {
  height: 100vh !important; /* The !important flag ensures this style overrides any other conflicting styles */
}

#component-0 {
  height: 100%;
}

#chatbot {
  flex-grow: 1; /* Allows the chatbot to expand to fill available space */
  overflow: auto; /* Adds a scrollbar if the content overflows the element's box */
}
"""


with gr.Blocks(css=CSS) as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id='chatbot',
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "tests/example.png"))),
    )

    with gr.Row():
        button = gr.UploadButton("LLM Augment 📁", file_types=['image'])
        fogButton = gr.UploadButton("Fog Augment 🌫️", file_types=['image'])
        autoButton = gr.UploadButton("Auto Augment 🤖", file_types=['image'])

    file_message = button.upload(add_file, [chatbot, button], [chatbot], queue=False).then(
      bot, chatbot, chatbot
    )
    fog_message = fogButton.upload(add_file, [chatbot, fogButton], [chatbot], queue=False).then(
      fogBot, chatbot, chatbot
    )
    auto_message = autoButton.upload(add_file, [chatbot, autoButton], [chatbot], queue=False).then(
      autoBot, chatbot, chatbot
    )


demo.launch(share=False)

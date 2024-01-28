from brancher import BranchingAgent
import gradio as gr
import os
import time
from PIL import Image

agent = BranchingAgent()
images = {}
tmpdir: str = ""
def add_file(messages, file):
    """
    """
    print(messages)
    images[file.name] = Image.open(file)
    messages = messages + [((file.name,), None)]
    global tmpdir
    tmpdir = os.path.dirname(file.name)
    yield messages

counter = 0
def new_image_path() -> str:
    global counter
    counter += 1
    return os.path.join(tmpdir, f".__wzhao6_internal__{counter}.png")

def bot(messages):
    """
    """
    print(messages)
    messages[-1][1] = "**Generating augmentations**"
    branches = agent.branch(images[messages[-1][0][0]])
    names = []
    for output in branches:
      path = new_image_path()
      output.save(path)
      names.append(os.path.basename(path))
    messages.append(
      (names, None)
    )

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
        button = gr.UploadButton("📁", file_types=['image'])

    file_message = button.upload(add_file, [chatbot, button], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )


demo.launch(share=False)

import gradio as gr
import os
import time


def add_file(messages, file):
    """
    """
    print(messages)
    messages = messages + [((file.name,), None)]
    return messages


def bot(messages):
    """
    """
    response = "**This is a simple test scripting that is longer than usual and contains no useful content!**"
    messages[-1][1] = ""
    for character in response:
        messages[-1][1] += character
        time.sleep(0.01)
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
        button = gr.UploadButton("📁", file_types=['image'])

    file_message = button.upload(add_file, [chatbot, button], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )


demo.launch()

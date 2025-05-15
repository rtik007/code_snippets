import ipywidgets as widgets
from IPython.display import display
from your_module import connect       # ← your real import
from your_config_module import GCP_CONFIG  # ← where GCP_CONFIG lives


# 2) build your UI
question_box = widgets.Textarea(
    placeholder='Type your question here…',
    layout=widgets.Layout(width='100%', height='80px')
)
submit_button = widgets.Button(description='Submit', button_style='success')

# 3) Markdown widget for the response
response_md = widgets.Markdown(
    value="", 
    layout=widgets.Layout(
        border='1px solid #ddd',
        padding='12px',
        margin='10px 0'
    )
)

# 4) when you click “Submit”, fetch & dump straight into Markdown
def on_submit(_):
    q = question_box.value.strip()
    if not q:
        response_md.value = "**Please enter a question.**"
        return

    raw = my_endpoint.generate(q)
    response_md.value = raw  # <-- raw is assumed to be Markdown

submit_button.on_click(on_submit)

# 5) display everything
display(widgets.VBox([question_box, submit_button, response_md]))

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript

# Input and submit UI
question_box = widgets.Textarea(
    value='',
    placeholder='Type your question here...',
    description='Question:',
    layout=widgets.Layout(width='100%', height='100px')
)
submit_button = widgets.Button(description='Submit', button_style='success')
output = widgets.Output()

# HTML template with collapsible section + copy button
def format_response_html(response):
    escaped_response = response.replace("<", "&lt;").replace(">", "&gt;")  # Safe escaping
    return f"""
    <style>
        .response-box {{
            background-color: #f0f4f8;
            border-left: 6px solid #4CAF50;
            padding: 15px;
            font-family: Arial, sans-serif;
            margin-top: 10px;
            position: relative;
        }}
        .copy-button {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 4px 8px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 12px;
        }}
        .copy-button:hover {{
            background-color: #45a049;
        }}
    </style>
    <div class="response-box">
        <button class="copy-button" onclick="navigator.clipboard.writeText(document.getElementById('response-text').innerText)">Copy</button>
        <details open>
            <summary><strong>LLM Response (click to expand/collapse)</strong></summary>
            <pre id="response-text" style="white-space: pre-wrap; line-height: 1.6;">{escaped_response}</pre>
        </details>
    </div>
    """

# Submission function
def on_submit(b):
    with output:
        output.clear_output()
        question = question_box.value.strip()
        if question:
            response = my_endpoint.generate(question)
            display(HTML(format_response_html(response)))
        else:
            display(HTML("<b style='color:red;'>Please enter a question.</b>"))

# Bind the button
submit_button.on_click(on_submit)

# Show UI
display(widgets.VBox([question_box, submit_button, output]))

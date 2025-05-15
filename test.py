import ipywidgets as widgets
from IPython.display import Markdown, display, HTML
from your_module import connect
from your_config_module import GCP_CONFIG
import time

# Add a function to reconnect if needed
def get_connection():
    try:
        # Try to use existing connection if it exists and is valid
        if hasattr(get_connection, 'endpoint') and get_connection.endpoint:
            # You might want to add a ping/health check here if your API supports it
            return get_connection.endpoint
    except Exception:
        # Connection expired or invalid
        pass
    
    # Create new connection
    get_connection.endpoint = connect('gemini-2.0-flash-001', config=GCP_CONFIG)
    return get_connection.endpoint

# Initial connection
get_connection.endpoint = connect('gemini-2.0-flash-001', config=GCP_CONFIG)

# Build improved UI
question_box = widgets.Textarea(
    placeholder='Type your question…',
    layout=widgets.Layout(width='100%', height='80px')
)
submit_button = widgets.Button(
    description='Submit',
    button_style='success',
    icon='paper-plane'  # Add an icon for better UX
)
status = widgets.HTML(
    value="",
    layout=widgets.Layout(margin='5px 0')
)
output = widgets.Output(
    layout=widgets.Layout(
        border='1px solid #ddd',
        padding='15px',
        margin='10px 0',
        min_height='150px'
    )
)

# Character counter
char_counter = widgets.HTML(
    value="<span style='color:#666'>0 characters</span>",
    layout=widgets.Layout(margin='5px 0')
)

# Update character count as user types
def update_char_count(_):
    count = len(question_box.value)
    color = "#666" if count < 1000 else "#e67e22" if count < 2000 else "#e74c3c"
    char_counter.value = f"<span style='color:{color}'>{count} characters</span>"

question_box.observe(update_char_count, names='value')

# Improved handler with error handling and loading state
def on_submit(_):
    # Get the question
    q = question_box.value.strip()
    
    # Clear previous outputs
    output.clear_output()
    
    # Validate input
    if not q:
        with output:
            display(Markdown("**Please enter a question.**"))
        return
    
    # Update status to show loading
    submit_button.disabled = True
    status.value = "<span style='color:#3498db'>Processing request...</span>"
    
    # Use try-except for robust error handling
    try:
        start_time = time.time()
        
        # Get connection (will reconnect if needed)
        endpoint = get_connection()
        
        # Call the model
        with output:
            raw = endpoint.generate(q)
            display(Markdown(raw))
        
        # Show success and timing
        elapsed = time.time() - start_time
        status.value = f"<span style='color:#2ecc71'>Completed in {elapsed:.2f} seconds</span>"
        
    except Exception as e:
        # Handle errors gracefully
        with output:
            error_message = f"**Error:** {str(e)}"
            display(Markdown(error_message))
        status.value = "<span style='color:#e74c3c'>An error occurred</span>"
    
    finally:
        # Re-enable the button
        submit_button.disabled = False

# Connect the handler
submit_button.on_click(on_submit)

# Create a clear button
clear_button = widgets.Button(
    description='Clear',
    button_style='warning',
    icon='eraser'
)

def on_clear(_):
    question_box.value = ""
    output.clear_output()
    status.value = ""
    
clear_button.on_click(on_clear)

# Create button layout
button_box = widgets.HBox([submit_button, clear_button])

# Show the complete UI
display(widgets.VBox([
    widgets.HTML("<h3>Gemini 2.0 Flash Interface</h3>"),
    widgets.Label("Enter your question below:"),
    question_box,
    char_counter,
    button_box,
    status,
    output
]))

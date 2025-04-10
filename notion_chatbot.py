import gradio as gr
import asyncio
import logging
from src.notion_agent import create_notion_agent, process_message

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def respond(message, history):
    """
    Process a user message and update chat history.
    
    Args:
        message: User's input message
        history: Current chat history
        
    Returns:
        Updated chat history with new message pair
    """
    try:
        # Initialize agent on first message
        if not hasattr(respond, 'agent'):
            respond.agent, respond.client, respond.memory = await create_notion_agent()
        
        # Process message and update history
        response = await process_message(message, respond.agent, respond.memory)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_msg = f"❌ An error occurred: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history

def clear_memory():
    """Clear the agent's memory and chat history."""
    if hasattr(respond, 'memory'):
        respond.memory.clear_memory()
    return None

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Notion Chatbot")
    gr.Markdown("Ask questions about your Notion workspace or request actions to be performed.")
    
    chatbot = gr.Chatbot(height=500, type='messages')
    msg = gr.Textbox(
        placeholder="Type your message here...",
        lines=2,
        show_label=False,
        container=False
    )
    clear = gr.Button("Clear")
    
    msg.submit(respond, [msg, chatbot], [chatbot])
    clear.click(clear_memory, None, [chatbot], queue=False)

if __name__ == "__main__":
    async def main():
        try:
            demo.queue().launch(
                server_name="127.0.0.1",
                server_port=7860,
                show_error=True,
                share=False
            )
        except KeyboardInterrupt:
            if hasattr(respond, 'client'):
                await respond.client.__aexit__(None, None, None)
    
    asyncio.run(main()) 
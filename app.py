import streamlit as st
import asyncio
import os
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

# --- 1. Streamlit App Configuration ---
st.set_page_config(
    page_title="ADK Market Researcher",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Google ADK Market Researcher")
st.caption("Powered by Google Agent Development Kit (ADK), LiteLLM, and Gemma 2 via Ollama")

# --- 2. Configuration & State Management ---
# Allow the endpoint to be overridden via environment variable (useful for Render deployment)
DEFAULT_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "https://evident-lens-surpass.ngrok-free.dev")

with st.sidebar:
    st.header("Pipeline Configurations")
    ollama_url = st.text_input("Ollama Base URL", value=DEFAULT_ENDPOINT)
    model_name = st.text_input("Model Name", value="ollama_chat/gemma2:2b")
    st.markdown("---")
    st.markdown("**Status:** Initializing backend environment...")

# Initialize session state for keeping track of messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Custom Search Tool & ADK Component Instantiation ---
class DDGHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.in_snippet = False
        self.current_snippet = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'td' and attrs_dict.get('class') == 'result-snippet':
            self.in_snippet = True

    def handle_data(self, data):
        if self.in_snippet:
            self.current_snippet.append(data.strip())

    def handle_endtag(self, tag):
        if tag == 'td' and self.in_snippet:
            snippet_text = " ".join(self.current_snippet).strip()
            if snippet_text:
                self.results.append(snippet_text)
            self.in_snippet = False
            self.current_snippet = []

def custom_web_search(query: str) -> str:
    """Searches the internet for up-to-date real-time data, current market trends, or recent news events."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            html_content = response.read().decode('utf-8')
            
        parser = DDGHTMLParser()
        parser.feed(html_content)
        snippets = parser.results[:3]
        
        if not snippets:
            return "Search completed but no relevant text summaries were found."
        return "\n---\n".join([f"[Result]: {text}" for text in snippets])
    except Exception as e:
        return f"Search execution encountered an error: {str(e)}"

# Setup ADK framework components
local_gemma = LiteLlm(model=model_name, api_base=ollama_url)
adk_search_tool = FunctionTool(custom_web_search)

research_agent = Agent(
    name="market_researcher",
    description="An agent capable of browsing live internet data to answer user queries accurately.",
    instruction=(
        "We are an expert research team analyzing financial market data. Use the custom_web_search tool "
        "to retrieve up-to-date details. Extract facts cleanly and provide a brief synthesis."
    ),
    model=local_gemma,
    tools=[adk_search_tool] 
)

APP_NAME = "market_research_app"
USER_ID = "user_streamlit"
SESSION_ID = "session_streamlit"

runner = InMemoryRunner(agent=research_agent, app_name=APP_NAME)

# --- 4. Core Async Execution Engine ---
async def execute_adk_agent(query_text: str, placeholder):
    """Wrapper function to execute the async ADK pipeline stream inside Streamlit."""
    # Ensure the session structure is initialized inside the runner service
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    
    content = types.Content(role='user', parts=[types.Part(text=query_text)])
    full_response = ""
    
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
            if hasattr(event, 'content') and event.content:
                if hasattr(event.content, 'parts') and event.content.parts:
                    chunk = event.content.parts[0].text
                    full_response += chunk
                    # Stream chunks dynamically to the user interface
                    placeholder.markdown(full_response + "▌")
            
            if hasattr(event, 'is_final_response') and event.is_final_response():
                break
    except Exception as e:
        st.error(f"Execution pipeline failed: {str(e)}")
        return f"Error: {str(e)}"
        
    placeholder.markdown(full_response)
    return full_response

# --- 5. UI Rendering & Interactions ---
# Display historical message board
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt handling
if user_query := st.chat_input("Ask about market trends (e.g., Nifty 500 trends this week)..."):
    # Append and show user query immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Render AI assistant stream framework
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Agent routing queries and checking web updates..."):
            # Execute async pipeline loop in sync environment
            ai_final_text = asyncio.run(execute_adk_agent(user_query, response_placeholder))
            
    # Save assistant response state
    st.session_state.messages.append({"role": "assistant", "content": ai_final_text})

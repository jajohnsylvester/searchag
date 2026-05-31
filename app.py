import streamlit as st
import asyncio
import os
import json
import logging
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from contextlib import aclosing
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
import litellm

# --- 1. Streamlit App & Log Configuration ---
st.set_page_config(
    page_title="ADK Market Researcher",
    page_icon="📈",
    layout="wide"
)

# Mute noisy internal OpenTelemetry context warnings from flooding the UI or logs
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

# Force LiteLLM to handle structural tool fallbacks via prompt injection for smaller models
litellm.add_to_system_prompt_failed_tool_calls = True

st.title("📈 Google ADK Market Researcher")
st.caption("Powered by Google Agent Development Kit (ADK) with Dynamic Local Model Fetching")

# --- 2. Dynamic Ollama Model Fetcher Logic ---
DEFAULT_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "https://evident-lens-surpass.ngrok-free.dev")

@st.cache_data(ttl=30)  # Caches the available models list for 30 seconds
def fetch_available_ollama_models(base_url: str) -> list:
    """Queries the remote/local Ollama API tags endpoint to fetch downloaded models."""
    fallback_models = ["ollama_chat/llama3.2", "ollama_chat/llama3.1", "ollama_chat/gemma2:9b"]
    try:
        # Clean up the URL formatting string to handle the native endpoints cleanly
        clean_url = base_url.rstrip('/')
        api_url = f"{clean_url}/api/tags"
        
        req = urllib.request.Request(api_url, headers={'User-Agent': 'Streamlit-Fetch'})
        with urllib.request.urlopen(req, timeout=4) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                models = data.get("models", [])
                if models:
                    # Map the raw model tags into LiteLLM-compatible format strings
                    return [f"ollama_chat/{m['name']}" for m in models]
    except Exception as e:
        # Silently log error to sidebar instead of crashing the main thread loop
        pass
    return fallback_models

# --- 3. Sidebar UI Layout Configurations ---
with st.sidebar:
    st.header("Pipeline Configurations")
    ollama_url = st.text_input("Ollama Base URL", value=DEFAULT_ENDPOINT)
    
    # 🚀 AUTOMATIC LOADING: Query the target endpoint and render inside selectbox
    with st.spinner("Scanning Ollama endpoint for available local models..."):
        available_models = fetch_available_ollama_models(ollama_url)
        
    selected_model = st.selectbox(
        "Select Active Local Model", 
        options=available_models,
        index=0
    )
    
    st.markdown("---")
    st.markdown(f"**Connected Model:** `{selected_model}`")
    if st.button("🔄 Refresh Model List"):
        st.cache_data.clear()
        st.sidebar.success("Model cache flushed!")

# Initialize chat session message board memory structures
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. Pure Python Custom Search Tool ---
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
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
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

# Setup dynamic ADK components relative to sidebar dropdown parameters
local_llm_backend = LiteLlm(model=selected_model, api_base=ollama_url)
adk_search_tool = FunctionTool(custom_web_search)

research_agent = Agent(
    name="market_researcher",
    description="An agent capable of browsing live internet data to answer user queries accurately.",
    instruction=(
        "We are an expert research team analyzing financial market data. Use the custom_web_search tool "
        "to retrieve up-to-date details. Extract facts cleanly and provide a brief synthesis."
    ),
    model=local_llm_backend,
    tools=[adk_search_tool] 
)

APP_NAME = "market_research_app"
USER_ID = "user_streamlit"
SESSION_ID = "session_streamlit"

runner = InMemoryRunner(agent=research_agent, app_name=APP_NAME)

# --- 5. Thread-Safe Async Execution Engine ---
async def execute_adk_agent(query_text: str, placeholder):
    """Wrapper function to execute the async ADK pipeline stream inside Streamlit."""
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    
    content = types.Content(role='user', parts=[types.Part(text=query_text)])
    full_response = ""
    
    try:
        # Wrap generator in aclosing context manager to handle pipeline detachment correctly
        async with aclosing(runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content)) as async_generator:
            async for event in async_generator:
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts') and event.content.parts:
                        chunk = event.content.parts[0].text
                        
                        if chunk is not None:
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                
                if hasattr(event, 'is_final_response') and event.is_final_response():
                    break
    except Exception as e:
        if "was created in a different Context" not in str(e):
            st.error(f"Execution pipeline failed: {str(e)}")
            return f"Error: {str(e)}"
        
    if not full_response:
        full_response = "The selected model executed your task but didn't print visible text output tokens."
        
    placeholder.markdown(full_response)
    return full_response

# --- 6. Chat UI Display & Execution Router ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about market trends..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner(f"Agent routing query using {selected_model}..."):
            ai_final_text = asyncio.run(execute_adk_agent(user_query, response_placeholder))
            
    st.session_state.messages.append({"role": "assistant", "content": ai_final_text})

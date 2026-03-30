"""
Streamlit chat application for the Evidently RAG agent.

Usage:
    streamlit run project/src/app.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure project/ is on the Python path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.agent import create_agent, log_interaction_to_file

st.set_page_config(page_title="Evidently AI Assistant", page_icon="📚")
st.title("Evidently AI Assistant")


@st.cache_resource
def get_agent():
    return create_agent()


agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about Evidently..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            result = asyncio.run(agent.run(user_prompt=user_input))
            st.markdown(result.output)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.output}
    )

    log_interaction_to_file(agent, result.new_messages())

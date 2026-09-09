import logging

import streamlit as st

from src.rag_chain import answer_question
from src.vector_store import get_vector_store


logging.basicConfig(level=logging.INFO)


@st.cache_resource(show_spinner="Initializing the SafeX knowledge base...")
def load_vector_store():
    """Create or load the Chroma store once per Streamlit process."""
    return get_vector_store()


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="SafeX Support",
    page_icon="💬"
)

st.title("SafeX Customer Support")

st.caption(
    "Ask questions about SafeX orders, shipping, returns, payments, and accounts."
)


# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            st.caption(
                "Sources: " + ", ".join(message["sources"])
            )


# -----------------------------
# Clear Chat
# -----------------------------

if st.button("Clear chat"):

    st.session_state.messages = []

    st.rerun()


try:
    store = load_vector_store()
except (FileNotFoundError, ValueError, RuntimeError) as error:
    logging.exception("Knowledge-base initialization failed")
    st.error(f"The knowledge base could not be initialized: {error}")
    st.stop()

question = st.chat_input(
    "Ask a SafeX support question..."
)


if question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):

        try:
            result = answer_question(question, vector_store=store)
            st.markdown(result.answer)

            if result.sources:
                st.caption("Sources: " + ", ".join(result.sources))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                    }
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                    }
                )


        except Exception as error:

            logging.exception(
                "Chat request failed"
            )

            st.error(
                f"The chatbot could not process that request: {error}"
            )

            st.caption(
                "Technical details were written to the terminal "
                "running Streamlit."
            )
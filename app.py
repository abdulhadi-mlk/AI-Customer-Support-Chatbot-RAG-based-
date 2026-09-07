import logging

import streamlit as st

from src.config import PROJECT_ROOT
from src.rag_chain import answer_question
from src.retriever import retrieve
from src.vector_store import get_vector_store


logging.basicConfig(level=logging.INFO)


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


# -----------------------------
# Chat Input
# -----------------------------

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

            # Check if ChromaDB exists
            if not (PROJECT_ROOT / "chroma_db").exists():

                st.error(
                    "The vector database is missing. "
                    "Run `python -m src.vector_store` first."
                )

            else:

                # -----------------------------
                # Get ChromaDB Vector Store
                # -----------------------------

                store = get_vector_store()


                # -----------------------------
                # Retrieval Debug Test
                # -----------------------------

                test_result = retrieve(
                    store,
                    question,
                    k=5
                )

                print("\n========== RETRIEVAL TEST ==========")

                print("Question:", question)

                print(
                    "Best score:",
                    test_result.best_score
                )

                print(
                    "Documents found:",
                    len(test_result.documents)
                )


                for i, doc in enumerate(
                    test_result.documents
                ):

                    print(
                        f"\n--- Document {i + 1} ---"
                    )

                    print(
                        doc.page_content[:1000]
                    )


                print(
                    "====================================\n"
                )


                # -----------------------------
                # Generate Final Answer
                # -----------------------------

                result = answer_question(
                    question,
                    vector_store=store
                )


                # -----------------------------
                # Display Answer
                # -----------------------------

                st.markdown(result.answer)


                # -----------------------------
                # Display Sources
                # -----------------------------

                if result.sources:

                    st.caption(
                        "Sources: "
                        + ", ".join(result.sources)
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources
                        }
                    )

                else:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer
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
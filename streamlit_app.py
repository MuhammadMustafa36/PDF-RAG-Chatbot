
# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

# Import Streamlit for creating the web interface.
import streamlit as st

# Import our RAG question-answering function.
from rag import ask_rag


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

# Configure the Streamlit page.
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)


# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------

# Display the main title.
st.title("📚 PDF RAG Chatbot")

# Display a short description.
st.write(
    "Ask questions about your PDF and get answers "
    "with source and page citations."
)


# ---------------------------------------------------------
# CHAT MEMORY
# ---------------------------------------------------------

# Streamlit reruns the script whenever the user interacts
# with the page.
#
# Therefore, we use session_state to remember chat messages.

if "messages" not in st.session_state:

    # Create an empty list for chat messages.
    st.session_state.messages = []


# ---------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------------------------------

# Go through all previous messages.
for message in st.session_state.messages:

    # Create a chat message container.
    with st.chat_message(message["role"]):

        # Display the message text.
        st.markdown(message["content"])


# ---------------------------------------------------------
# CLEAR CHAT BUTTON
# ---------------------------------------------------------

# Create a button in the sidebar.
if st.sidebar.button("🗑️ Clear Chat"):

    # Remove all previous messages.
    st.session_state.messages = []

    # Refresh the page.
    st.rerun()


# ---------------------------------------------------------
# USER QUESTION
# ---------------------------------------------------------

# Create the chat input box.
question = st.chat_input(
    "Ask a question about your PDF..."
)


# ---------------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------------

# Check whether the user entered a question.
if question:

    # -----------------------------------------
    # DISPLAY USER QUESTION
    # -----------------------------------------

    # Add the user's question to chat history.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Display the user's message.
    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------
    # GENERATE RAG ANSWER
    # -----------------------------------------

    # Show the assistant response.
    with st.chat_message("assistant"):

        # Display a loading message while the model works.
        with st.spinner("Searching the PDF and generating answer..."):

            # Send the question to our RAG system.
            answer, citations = ask_rag(question)


        # -----------------------------------------
        # DISPLAY ANSWER
        # -----------------------------------------

        # Display the generated answer.
        st.markdown(answer)


        # -----------------------------------------
        # DISPLAY SOURCES
        # -----------------------------------------

        if citations:

            st.markdown("### 📚 Sources")

            # Display each source and page.
            for source, page in citations:

                st.write(
                    f"- `{source}` — Page {page}"
                )


        # -----------------------------------------
        # SAVE ASSISTANT RESPONSE
        # -----------------------------------------

        # Save the answer in chat history.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


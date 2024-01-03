import os
import streamlit as st
from retrievalService import RetrievalService

def main():
    st.title("Github Bot")

    # Set up the retrieval service
    @st.cache_resource
    def get_retrieval_service(github_link):
        retrieval_service = RetrievalService(github_link)
        retrieval_service.getting_repo()
        retrieval_service.get_docs()
        retrieval_service.embedding()
        retrieval_service.retrieval()
        retrieval_service.delete_directory()
        return retrieval_service

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User input for GitHub repository link
    user_repo = st.text_input("GitHub Link to your public codebase")
    if user_repo:
        retrieval_service = get_retrieval_service(user_repo)

    # Model selection buttons
    st.write("Choose a model for further interactions:")
    model_options = ["llama70B", "Mixtral 7x8", "zephyr7b"]
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_options[2]

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("llama70B"):
            st.session_state.selected_model = "llama70B"
    with col2:
        if st.button("Mixtral 7x8"):
            st.session_state.selected_model = "Mixtral 7x8"
    with col3:
        if st.button("zephyr7b"):
            st.session_state.selected_model = "zephyr7b"

    st.write(f"Current model: {st.session_state.selected_model}")

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input (question)
    if prompt := st.chat_input("Type your question here."):
        if prompt:  # Check for non-empty prompt
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            conversation_chain = retrieval_service.conversation(st.session_state.selected_model)
            # Show a loading animation while retrieving the answer
            with st.spinner("Retrieving the answer..."):
                try:
                    # Retrieve answer from the chatbot
                    response = conversation_chain(prompt)
                except Exception as e:
                    response = f"Error in processing your question: {e}"

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()

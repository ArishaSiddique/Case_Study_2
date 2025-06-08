import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama  # using local LLM
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

# this function reads all the text from the uploaded pdfs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# breaks the big text into smaller chunks so model can handle easily
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# this creates a vector store (for similarity search) from the chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# main logic to make the chatbot work using local model through ollama
def get_conversation_chain(vectorstore):
    llm = Ollama(model="mistral")  # you can change model here if needed

    # this keeps track of the chat history
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    # setting up the whole question-answering flow
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# this shows the chat messages on the screen
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process documents first.")
        return

    try:
        # get response from chatbot
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # show chat history (user + bot)
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Something went wrong in the chat: {e}")

# main streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # setting up session stuff
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    # input box to ask questions
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # file upload section on the left
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    try:
                        # step-by-step: read → chunk → embed → chatbot
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(vectorstore)

                        if st.session_state.conversation:
                            st.success("Done! Ask your questions now.")
                        else:
                            st.error("Something went wrong while setting up.")
                    except Exception as e:
                        st.error(f"Error while processing: {e}")


if __name__ == '__main__':
    main()

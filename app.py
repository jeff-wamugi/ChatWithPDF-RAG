import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and get a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load FAISS index with dangerous deserialization enabled
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    # Get the response from the chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main application
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using Gemini - RAG 🕵🏻️")

    # Initialize session state for questions and answers
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    # Text input for user question
    user_question = st.text_input("Ask a Question from the PDF Files You Submitted")

    if user_question:
        # Process the question and get the response
        response = user_input(user_question)
        
        # Store the Q&A in session state for history
        st.session_state.qa_history.append(("Q: " + user_question, "A: " + response))

    # Display the Q&A history
    if st.session_state.qa_history:
        for qa in st.session_state.qa_history:
            st.write(qa[0])  # Display question
            st.write(qa[1])  # Display answer

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
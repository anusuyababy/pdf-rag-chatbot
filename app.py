
import os
import tempfile

import streamlit as st
from groq import Groq
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Load API key from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Streamlit page setup
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions from it.")


# Check API key
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please add it inside your .env file.")
    st.stop()


# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])


if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    st.success("PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    

    # Split PDF into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    

    # Create embeddings
    with st.spinner("Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

    # Create vector database
    with st.spinner("Creating vector database..."):
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

    # User question
    query = st.text_input("Ask a question from the PDF")

    if query:

        # Retrieve relevant chunks
        retrieved_docs = db.similarity_search(query, k=3)

        # Create context from retrieved chunks
        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        # Prompt for LLM
        prompt = f"""
You are a helpful PDF question-answering assistant.

Use only the context below to answer the question.
If the answer is not in the context, say:
"I don't know from the PDF."

Context:
{context}

Question:
{query}

Give a short and clear answer.
"""

        # Groq client
        client = Groq(api_key=groq_api_key)

        # Generate final answer
        with st.spinner("Generating answer..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

        answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        

else:
    st.info("Please upload a PDF to begin.")
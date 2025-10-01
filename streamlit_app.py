import streamlit as st
import os

import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# New imports (✅)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="PDF Q&A - Gemini", layout="wide")
st.title("📘 Ask Questions from Your PDF using Gemini")

st.markdown("""
This app lets you upload a PDF, processes it into vector embeddings, and lets you 
ask questions from your document using **Google Gemini (`gemini-flash-lite-latest`)**.
""")

# -----------------------
# 1️⃣ API Key Input
# -----------------------
api_key = st.text_input("Enter your Google API Key:", type="password")
if not api_key:
    st.warning("Please enter your Google API key to continue.")
    st.stop()

# Configure Gemini with user API key
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure Gemini: {e}")
    st.stop()

# -----------------------
# 2️⃣ File Upload
# -----------------------
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file is None:
    st.info("Please upload a PDF to start.")
    st.stop()

# Save the uploaded file temporarily
temp_path = "uploaded_document.pdf"
with open(temp_path, "wb") as f:
    f.write(uploaded_file.read())

# -----------------------
# 3️⃣ Load & Split Document
# -----------------------
try:
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    st.success("✅ PDF loaded successfully!")
except Exception as e:
    st.error(f"Error loading PDF: {e}")
    st.stop()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)
st.write(f"Document split into {len(texts)} chunks.")

# -----------------------
# 4️⃣ Create Vector Store
# -----------------------
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    st.success("✅ FAISS index created successfully!")
except Exception as e:
    st.error(f"Error creating FAISS index: {e}")
    st.stop()

# -----------------------
# 5️⃣ Question Answering
# -----------------------
st.subheader("Ask your question")
user_query = st.text_input("Enter your question about the PDF:")

if user_query:
    try:
        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(user_query, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        # Prepare prompt for Gemini
        prompt = f"""
        You are a helpful assistant. Use the following context from the document to answer the question.
        If the answer is not found, say "The answer is not available in the document."

        Context:
        {context}

        Question: {user_query}
        """

        # Query Gemini
        model = genai.GenerativeModel("gemini-flash-lite-latest")
        response = model.generate_content(prompt)

        # Display response
        st.markdown("### 🧠 Answer:")
        st.write(response.text)
    except Exception as e:
        st.error(f"An error occurred while generating the answer: {e}")

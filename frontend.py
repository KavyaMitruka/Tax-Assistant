import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm_memory import fetch_pdf_documents, save_uploaded_pdf, generate_text_segments, initialize_embedding_model

# Load environment token
ACCESS_TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPOSITORY = "mistralai/Mistral-7B-Instruct-v0.3"

def initialize_llm(repo_identifier):
    language_model = HuggingFaceEndpoint(
        repo_id=repo_identifier,
        temperature=0.5,
        model_kwargs={"token": ACCESS_TOKEN, "max_length": "512"}
    )
    return language_model

# Define a custom prompt format
PROMPT_TEMPLATE = """
Utilize the provided contextual information to answer the user's question.
If the answer isn't available in the context, state that you don't know it.
Avoid fabricating responses. Provide answers strictly based on the given context.

Context: {context}
Question: {question}

Respond concisely without unnecessary introductions.
"""

def create_prompt(prompt_structure):
    formatted_prompt = PromptTemplate(template=prompt_structure, input_variables=["context", "question"])
    return formatted_prompt

# Load the FAISS database safely
VECTOR_DB_PATH = "vectorstore/db_faiss"
INDEX_FILE = os.path.join(VECTOR_DB_PATH, "index.faiss")

embedding_processor = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(INDEX_FILE):
    vector_database = FAISS.load_local(VECTOR_DB_PATH, embedding_processor, allow_dangerous_deserialization=True)
else:
    st.warning("FAISS index not found. The assistant may not have relevant knowledge until documents are processed.")
    vector_database = None  # Handle missing FAISS index gracefully

# Streamlit UI setup
st.set_page_config(page_title="AI TaxBuddy", layout="wide")

with st.sidebar:
    st.title("AI TaxBuddy")
    option = st.radio("Navigation", ["Instructions", "Why Trust AI TaxBuddy?", "Assistant"])

if option == "Instructions":
    st.header("Instructions")
    st.write("""
    - Ensure you have all necessary financial documents before asking queries.
    - Be specific in your questions for better accuracy.
    - AI TaxBuddy provides insights, but always consult a tax professional before making financial decisions.
    - Use the Assistant tab to get AI-powered tax-related guidance.
    """)

elif option == "Why Trust AI TaxBuddy?":
    st.header("Why Trust AI TaxBuddy?")
    st.write("""
    - AI TaxBuddy is built on advanced AI models (Mistral-7B-Instruct) ensuring reliable tax insights.
    - Uses verified government tax data and retrieval-augmented generation (RAG) for accuracy.
    - Provides up-to-date information by integrating services for tax regulation updates.
    - Prioritizes security and transparency in handling financial queries.
    """)

elif option == "Assistant":
    st.header("Tax Assistant Chatbot")
    
    uploaded_files = st.file_uploader("Upload supporting Documents (optional)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            save_uploaded_pdf(file)  # Save the uploaded file first

        # Process and update FAISS DB
        docs = fetch_pdf_documents("data/")  # Corrected function call
        new_chunks = generate_text_segments(docs)
        emb_model = initialize_embedding_model()

        # Load or create FAISS vector store
        if os.path.exists(INDEX_FILE):
            db = FAISS.load_local(VECTOR_DB_PATH, emb_model, allow_dangerous_deserialization=True)
            db.add_documents(new_chunks)
        else:
            db = FAISS.from_documents(new_chunks, emb_model)

        db.save_local(VECTOR_DB_PATH)  # Save updated FAISS index
        st.success("Documents processed and FAISS index updated!")

    # Ensure FAISS database is loaded before querying
    if vector_database is None:
        st.error("No FAISS index found. Please upload documents first to enable tax assistance.")
    else:
        user_input = st.text_input("Enter your tax query:")
        if user_input:
            qa_pipeline = RetrievalQA.from_chain_type(
                llm=initialize_llm(MODEL_REPOSITORY),
                chain_type="stuff",
                retriever=vector_database.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': create_prompt(PROMPT_TEMPLATE)}
            )
            output_response = qa_pipeline.invoke({'query': user_input})
            st.write("Response:", output_response["result"])

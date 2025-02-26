import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

# Define the directory for storing PDFs
DIRECTORY = "data/"

def fetch_pdf_documents(directory):
    pdf_loader = DirectoryLoader(directory, glob='*.pdf', loader_cls=PyPDFLoader)
    extracted_documents = pdf_loader.load()
    return extracted_documents

def save_uploaded_pdf(uploaded_file):
    file_destination = os.path.join(DIRECTORY, uploaded_file.name)
    with open(file_destination, "wb") as file:
        file.write(uploaded_file.getbuffer())

# Load and process PDF documents
document_collection = fetch_pdf_documents(directory=DIRECTORY)

# Function to create text chunks
def generate_text_segments(source_data):
    segmenter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    segmented_text = segmenter.split_documents(source_data)
    return segmented_text

# Load website content
web_loader = WebBaseLoader([
    "https://incometaxindia.gov.in/Pages/default.aspx", 
    "https://www.incometax.gov.in/iec/foportal/"
])
web_documents = web_loader.load()

# Generate text chunks for both sources
web_text_segments = generate_text_segments(source_data=web_documents)
pdf_text_segments = generate_text_segments(source_data=document_collection)

# Function to initialize the embedding model
def initialize_embedding_model():
    model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    return model

embedding_processor = initialize_embedding_model()

# Define FAISS storage path
FAISS_STORAGE_PATH = "vectorstore/db_faiss"

# Create and store vector embeddings
db = FAISS.from_documents(pdf_text_segments, embedding_processor)
db.add_documents(web_text_segments)
db.save_local(FAISS_STORAGE_PATH)

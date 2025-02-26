import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# Load the FAISS database
VECTOR_DB_PATH = "vectorstore/db_faiss"
embedding_processor = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_database = FAISS.load_local(VECTOR_DB_PATH, embedding_processor, allow_dangerous_deserialization=True)

# Establish the retrieval-based QA system
qa_pipeline = RetrievalQA.from_chain_type(
    llm=initialize_llm(MODEL_REPOSITORY),
    chain_type="stuff",
    retriever=vector_database.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': create_prompt(PROMPT_TEMPLATE)}
)

# Execute the query process
user_input = input("Enter your query: ")
output_response = qa_pipeline.invoke({'query': user_input})
print("Response: ", output_response["result"])

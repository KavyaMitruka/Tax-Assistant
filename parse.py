import os
import json
from langchain.document_loaders import PyPDFLoader
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# The OPENAI_API_KEY is now loaded solely from the .env file.
# Make sure your .env file contains a line like:
# OPENAI_API_KEY=your_actual_openai_api_key_here

# Import and initialize the ChatGPT model
from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Load the PDF using PyPDFLoader (which uses pypdf under the hood)
pdf_path = "data/sample2.pdf"  # Replace with the path to your PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()
pdf_text = "\n".join(doc.page_content for doc in documents)

# Define a Pydantic schema for the structured output
class Employee(BaseModel):
    name: str = Field(..., description="Employee's full name")
    pan: str = Field(..., description="Employee PAN")
    aadhaar: str | None = Field(None, description="Employee Aadhaar (optional)")
    address: str = Field(..., description="Employee address")
    email: str | None = Field(None, description="Employee email")
    mobile_number: str | None = Field(None, description="Employee mobile number")
    assessment_year: str = Field(..., description="Assessment year (e.g., 2022-23)")

class Employer(BaseModel):
    name: str = Field(..., description="Employer's name")
    address: str = Field(..., description="Employer's address")
    tan: str = Field(..., description="Employer TAN")
    pan: str = Field(..., description="Employer PAN")
    contact_email: str | None = Field(None, description="Employer contact email")
    contact_number: str | None = Field(None, description="Employer contact number")

class SalaryDetails(BaseModel):
    basic_salary: float = Field(..., description="Basic salary in INR")
    gross_salary: float = Field(..., description="Gross salary in INR")
    net_salary: float = Field(..., description="Net salary after deductions")
    taxable_salary: float = Field(..., description="Taxable salary in INR")

class Form16Data(BaseModel):
    employees: Employee
    employers: Employer
    salary_details: SalaryDetails

# Set up the Pydantic output parser with our schema
parser = PydanticOutputParser(pydantic_object=Form16Data)

# Create a prompt template that provides instructions for the output format
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in extracting structured financial data from Form 16 documents."),
    ("user", "Extract structured data from the following Form 16 document and return the result in JSON format matching the following instructions:\n\n{format_instructions}\n\nDocument:\n{pdf_text}")
])
formatted_prompt = prompt_template.format(
    pdf_text=pdf_text,
    format_instructions=parser.get_format_instructions()
)

# Use the ChatGPT model to generate the structured output
structured_output = model.invoke(formatted_prompt)

# Parse the LLM output using the Pydantic output parser
try:
    structured_data = parser.parse(structured_output.content)
    print(json.dumps(structured_data.dict(), indent=4))
except Exception as e:
    print("Error parsing output:", e)

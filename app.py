import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import faiss
import json
import re
from langchain.docstore.document import Document
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    st.error("GOOGLE_API_KEY environment variable is not set.")
    st.stop()
genai.configure(api_key=api_key)

# Step 1: Document Processing
def get_pdf_text(pdf_docs):
    text = ""
    with ThreadPoolExecutor() as executor:
        results = executor.map(extract_text_from_pdf, pdf_docs)
        for result in results:
            text += result
    return text

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Step 2: Information Extraction and Tagging
def extract_information(text_chunk):
    equipment_name = re.findall(r'(?i)equipment name:\s*([^\n]+)', text_chunk)
    domain = re.findall(r'(?i)domain:\s*([^\n]+)', text_chunk)
    model_numbers = re.findall(r'(?i)model number:\s*([^\n]+)', text_chunk)
    manufacturer = re.findall(r'(?i)manufacturer:\s*([^\n]+)', text_chunk)
    
    tags = {
        'equipment_name': equipment_name[0] if equipment_name else 'N/A',
        'domain': domain[0] if domain else 'N/A',
        'model_numbers': model_numbers[0] if model_numbers else 'N/A',
        'manufacturer': manufacturer[0] if manufacturer else 'N/A',
    }
    
    return tags

# Step 3: Vector Database Integration
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    text_embeddings = []
    metadata = []

    def process_chunk(chunk):
        tags = extract_information(chunk)
        vector = embeddings.embed_query(chunk)
        return vector, {"page_content": chunk, "metadata": tags}

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_chunk, text_chunks)
        for vector, meta in results:
            text_embeddings.append(vector)
            metadata.append(meta)
    
    # Convert text embeddings to numpy array
    vectors = np.array(text_embeddings).astype(np.float32)
    print(np.array(vectors))
    
    # Initialize FAISS index
    dimension = vectors.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    # Save the index and metadata
    faiss.write_index(index, "faiss_index.index")
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

# Step 4: Query Processing
def parse_query(query):
    equipment = re.findall(r'(?i)equipment:\s*([^\n]+)', query)
    model = re.findall(r'(?i)model:\s*([^\n]+)', query)
    manufacturer = re.findall(r'(?i)manufacturer:\s*([^\n]+)', query)
    
    parsed_query = {
        'equipment': equipment[0] if equipment else '',
        'model': model[0] if model else '',
        'manufacturer': manufacturer[0] if manufacturer else ''
    }
    return parsed_query

def user_input(user_question):
    parsed_query = parse_query(user_question)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
    
    # Load FAISS index
    index = faiss.read_index("faiss_index.index")

    # Load metadata
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Convert user query to vector
    query_vector = embeddings.embed_query(user_question)
    query_vector = np.array(query_vector).astype(np.float32)
    print(len(query_vector))
    # Perform similarity search
    distances, indices = index.search(query_vector.reshape(1, -1), k=5)
    
    # Create Document objects from metadata
    docs = [Document(page_content=metadata[i]["page_content"], metadata=metadata[i]["metadata"]) for i in indices[0]]
    
    # Use QA chain to generate response
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Step 5: Response Generation
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "answer is not available in the context." Don't provide a wrong answer.\n\n
    Context:\n{context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Step 6: System Integration
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Intelligent Document Processing and Query System")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        with st.spinner("Generating response..."):
            response = user_input(user_question)
            st.write("Reply: ", response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True,
            type="pdf",
            key="pdf_uploader"
        )
        if st.button("Submit & Process"):
            if pdf_docs and len(pdf_docs) <= 10:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            elif len(pdf_docs) > 10:
                st.warning("Please upload up to 10 PDF files at once.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()

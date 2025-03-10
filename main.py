import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
import json

try:
   
    from langchain_community.embeddings import HuggingFaceEmbeddings
except (ImportError, TypeError):

    from langchain.embeddings import HuggingFaceEmbeddings

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_ollama_response(prompt):
    url = "http://127.0.0.1:11434/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            response_data = response.json()
           
            if 'choices' in response_data:
                return response_data.get('choices', [{}])[0].get('text', "No response generated")
            elif 'text' in response_data:
                return response_data.get('text', "No response generated")
            else:
                return str(response_data)
        else:
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

def get_conversational_chain():
    prompt_template = """
   You are an expert AI assistant tasked with answering questions based strictly on the provided context.  

- Provide a **detailed and accurate** response using only the information available in the context.  
- If the answer **cannot be found** within the context, respond with:  
  **"The answer is not available in the provided context."**  
- Do **not** generate misleading or fabricated answers.  

### Context:  
{context}  

### Question:  
{question}  

### Answer:

    """
    
    def ollama_chain(inputs):
        context = "\n".join([doc.page_content for doc in inputs["input_documents"]])
        question = inputs["question"]
        
        prompt = prompt_template.format(context=context, question=question)
        response = get_ollama_response(prompt)
        
        return {"output_text": response}
    
    return ollama_chain

def user_input(user_question):
    try:
      
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question}
        )
        
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Ollama (llama3.2) 💬")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
import boto3
import streamlit as stl
import uuid
import os
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


)

client_s3 = boto3.client("s3")
BUCKET_NAME = 'cricket-rules-saurabh-genaiproj'
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def text_splitter(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, docs):
    vectorstore_faiss=FAISS.from_documents(docs, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    client_s3.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    client_s3.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

def main():
    stl.write("test admin")
    file = stl.file_uploader("Select a PDF file", type="pdf")
    if file is not None:
        unique_id = str(uuid.uuid4())
        stl.write(f"Generated Request ID: {unique_id}")
        pdf_file_name = f"{unique_id}.pdf"
        with open(pdf_file_name, "wb") as file_writer:
            file_writer.write(file.getvalue())
        pdf_loader = PyPDFLoader(pdf_file_name)
        pdf_pages = pdf_loader.load_and_split()
        stl.write(f"Number of Pages: {len(pdf_pages)}")

        splitted_docs = text_splitter(pdf_pages, 1000, 200)
        stl.write(f"Splitted Docs length: {len(splitted_docs)}")
        stl.write("Vector Store Creation")
        result = create_vector_store(unique_id, splitted_docs)
        
        if result:
            stl.write("Completed, PDF processed successfully")
        else:
            stl.write("Error!!")

if __name__ == "__main__":
    main()



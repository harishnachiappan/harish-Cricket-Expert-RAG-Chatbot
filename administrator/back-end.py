import boto3
import streamlit as stl
import uuid
import os


client_s3 = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

def text_splitter(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

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
        stl.write("===================")
        stl.write(splitted_docs[0])
        stl.write("===================")
        stl.write(splitted_docs[1])

if __name__ == "__main__":
    main()



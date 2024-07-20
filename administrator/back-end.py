'''This code builds the admin app that allows you to upload your pdf, 
split it into chunks, vectorize and save the embeddings in S3'''


import boto3
import boto3.session
import streamlit as stl
import uuid
import tempfile
import os
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


client_s3 = boto3.client("s3")
BUCKET_NAME = os.getenv('BUCKET_NAME') 

'''all access variables stored locally, store your respective keys in .env file of your directory 
and reference the same while creating and running your docker container'''

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
print(AWS_REGION)
print(AWS_ACCESS_KEY_ID)

'''ensure that you have access to the models for your region in AWS Console, 
make sure the user under IAM has Bedrock access to invoke models'''

bedrock_client = boto3.client(service_name="bedrock-runtime",region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def text_splitter(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def simple_tokenizer(text):
    return len(text.split())

def count_tokens(docs):  #function to count tokens
    total_tokens = 0
    for doc in docs:
        total_tokens += simple_tokenizer(doc.page_content)
    return total_tokens

def create_vector_store(request_id, docs):
    vectorstore_faiss=FAISS.from_documents(docs, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path = "./temp"
    os.makedirs(folder_path, exist_ok=True)
    try:
        vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
        print(f"Files saved locally in '{folder_path}'.")
    except Exception as e:
        print(f"Failed to save FAISS vector store locally: {e}")
        return False
    
    print("Contents after saving files:")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            print(os.path.join(root, file))
    client_s3.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    client_s3.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    return True

def main():
    stl.write("test admin")
    file = stl.file_uploader("Select a PDF file", type="pdf")
    '''
    At first, test the vectore creating with smaller context, i.e. try uploading a single page pdf and see
    the kind of results you get in the user app
    '''

    '''
    Research about pricing per input/output token for the model that you choose to use
    '''
    if file is not None:
        unique_id = str(uuid.uuid4())
        stl.write(f"Generated Request ID: {unique_id}")
        print("")
        pdf_file_name = f"{unique_id}.pdf"
        with open(pdf_file_name, "wb") as file_writer:
            file_writer.write(file.getvalue())
        pdf_loader = PyPDFLoader(pdf_file_name)
        pdf_pages = pdf_loader.load_and_split()
        stl.write(f"Number of Pages: {len(pdf_pages)}")

        splitted_docs = text_splitter(pdf_pages, 1000, 200)
        stl.write(f"Splitted Docs length: {len(splitted_docs)}")
        token_count = count_tokens(splitted_docs)
        stl.write(f"Estimated Total Tokens: {token_count}")
        stl.write("Vector Store Creation")
        result = create_vector_store(unique_id, splitted_docs)
        print(result)
        if result:
            stl.write("Completed, PDF processed successfully")
        else:
            stl.write("Error!!")

if __name__ == "__main__":
    main()



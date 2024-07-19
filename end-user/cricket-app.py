import boto3
import boto3.session
import streamlit as stl
import uuid
import tempfile

import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock


client_s3 = boto3.client("s3")
BUCKET_NAME = os.getenv('BUCKET_NAME')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')




bedrock_client = boto3.client(service_name="bedrock-runtime",region_name=AWS_REGION)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)
folder_path = "./temp"
os.makedirs(folder_path, exist_ok=True)

def get_index():
    try:
        client_s3.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}/my_faiss.faiss")
        stl.write("Successfully downloaded my_faiss.faiss")
    except Exception as e:
        stl.write(f"Failed to download my_faiss.faiss: {e}")

    try:
        client_s3.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}/my_faiss.pkl")
        stl.write("Successfully downloaded my_faiss.pkl")
    except Exception as e:
        stl.write(f"Failed to download my_faiss.pkl: {e}")


def get_llm():
    llm=Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock_client)
    return llm


def get_response(llm,vectorstore, question ):
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']

def main():
    stl.write("test end-user")

    get_index()

    dir_list = os.listdir(folder_path)
    stl.write(f"Files in {folder_path}")
    stl.write(dir_list)

    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    stl.write("INDEX IS READY")

    question = stl.text_input("Please ask your question")
    if stl.button("Ask Question"):
        with stl.spinner("Querying..."):
            llm = get_llm()

            input_text = f"User: {question}\nBot:"
            text_generation_config = {
                "temperature": 0.3,  # Default value, adjust if necessary
                "topP": 0.5,  # Default value, adjust if necessary
                "maxTokenCount": 500,
                "stopSequences": ["\n"]
            }

            # get_response
            stl.write(get_response(llm, faiss_index, input_text))
            stl.success("Done")


if __name__ == "__main__":
    main()



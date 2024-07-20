'''
This is the code that builds the fron-end user app, i.e. the functioning chatbot
'''



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


def get_index(): #retrieving indices saved in S3, make sure to add exceptions
    try:
        client_s3.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}/my_faiss.faiss")
        stl.write("Successfully downloaded my_faiss.faiss from S3 bucket")
    except Exception as e:
        stl.write(f"Failed to download my_faiss.faiss: {e}")
    try:
        client_s3.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}/my_faiss.pkl")
        stl.write("Successfully downloaded my_faiss.pkl from S3 bucket")
    except Exception as e:
        stl.write(f"Failed to download my_faiss.pkl: {e}")

'''
Research token pricing/input output before you choose the model. 
Note: You need to submit special request form for Anthropic models
'''
def get_llm():
    llm=Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock_client)
    return llm



def get_response(llm,vectorstore, question ):
    prompt_template = """
    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer to the question, communicate that the information that the user is seeking
    is not within your area of expertise and you are open to answering questions within given context
    <context>
    {context}
    </context>
    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
    return_source_documents=True,chain_type_kwargs={"prompt": PROMPT})
    answer=qa({"query":question})
    return answer['result']

def main():


    stl.set_page_config(layout='wide')
    stl.markdown(
        """
        <style>
        .stApp {
            background-color: white;
            font-family: 'Helvetica', sans-serif;
            font-weight: 300;
            color: black;
        }
        .stApp h1 {
            font-family: 'Helvetica', sans-serif;
            font-weight: 300;
            color: black;
        }
        .stTextInput > div > div > input {
            border: 2px solid black;
            color: white;
            background-color: black;
        }
        .stTextInput > div > div > input::placeholder {
            color: white;
        }
        .stTextInput > label {
            color: black;
        }
        .stButton > button {
            border: 2px solid black;
            color: white;
            background-color: black;
        }
        </style>
        """, unsafe_allow_html=True
    )



    stl.title("The Cricket Expert Chatbot by [Saurabh Sonawane](https://www.linkedin.com/in/saurabh112000/)")
    stl.write("Thank you for visiting the page! You can ask any question about Cricket and learn more about the game. Currently, the bot does not support query memory, so each question will have to be independent of the previous one. (Trying not to go broke :)) ")

    get_index()
    dir_list = os.listdir(folder_path)
    stl.write(f"Files stored in {folder_path}")
    stl.write(dir_list)

    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    stl.write("INDEX READY")
    question = stl.text_input("Please ask your question")
    if stl.button("Ask Question"):
        with stl.spinner("Querying..."):
            llm = get_llm()

            input_text = f"User: {question}\nBot:"
            text_generation_config = {
                "temperature": 0.9,  
                "topP": 0.2,  
                "maxTokenCount": 1000,
                "stopSequences": ["\n"]
            }

       
            stl.write(get_response(llm, faiss_index, input_text))
            stl.success("Done")


if __name__ == "__main__":
    main()



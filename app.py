"""A RAG QnA chain using chainlit"""
import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Typesense
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TYPESENSE_API_KEY = os.getenv('TYPESENSE_API_KEY')
PDF_DIRECTORY = "./content"

def load_pdf_directory():
    """loads all the pdfs in the a directory and returns the list"""
    documents = []
    for file in os.listdir(PDF_DIRECTORY):
        pdf_path = os.path.join(PDF_DIRECTORY, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents

def get_docs():
    """return chunk (split) docs, uses CharacterTextSplitter with chunk size = 1000 and overlap = 0 """
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = load_pdf_directory()
    docs = text_splitter.split_documents(documents)
    return docs

def get_embeddings():
    """Vector embeddings"""
    embeddings = OpenAIEmbeddings()
    return embeddings

def get_docsearch():
    """retriever using typesense"""
    docs = get_docs()
    embeddings = get_embeddings()
    docsearch = Typesense.from_documents(
        docs,
        embeddings,
        typesense_client_params={
            "host": "localhost",  # Use xxx.a1.typesense.net for Typesense Cloud
            "port": "8108",  # Use 443 for Typesense Cloud
            "protocol": "http",  # Use https for Typesense Cloud
            "typesense_api_key": TYPESENSE_API_KEY,
            "typesense_collection_name": "lang-chain",
        },
    )
    return docsearch

def get_retriever():
    """ returns the retriever"""
    docsearch = get_docsearch()
    retriever = docsearch.as_retriever()
    return retriever

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.8)

chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=get_retriever()
)
@cl.on_message
async def main(message: cl.Message):
    query = message.content
    response = chain(query) 
    await cl.Message(
        content=f"{response['result']}",
    ).send()

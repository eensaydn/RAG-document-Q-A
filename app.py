import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
import openai

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
if groq_api_key:
    os.environ['GROQ_API_KEY'] = groq_api_key

groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        embeddings = OpenAIEmbeddings()
        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs[:50])
        vectors = FAISS.from_documents(final_documents, embeddings)

        st.session_state.embeddings = embeddings
        st.session_state.loader = loader
        st.session_state.docs = docs
        st.session_state.text_splitter = text_splitter
        st.session_state.final_documents = final_documents
        st.session_state.vectors = vectors

st.title("RAG Document Q&A With Groq And Lama3")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')







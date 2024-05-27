import os
import pickle
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

logging.basicConfig(level=logging.INFO)

st.title("Chatgroq With Llama3 Application")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant designed to help answer questions based on the given context. Ensure your responses are accurate, concise, and directly related to the context provided.

<Context>
{context}
</Context>

Questions:
{input}

Response:

Example:
<Context>
In 2020, the global market for AI grew significantly due to increased investment in technology.
</Context>

Questions:
What factors contributed to the growth of the AI market in 2020?

Response:
The growth of the AI market in 2020 was primarily driven by increased investment in technology.
"""
)


@st.cache_resource
def vector_embedding():
    # enter the right .pkl file to make this work 
    embeddings_path = './embeddings.pkl'
    if os.path.exists(embeddings_path):
        logging.info('Loading precomputed embeddings')
        with open(embeddings_path, 'rb') as f:
            vectors = pickle.load(f)
    else:
        logging.info("Initialize embeddings")
        embeddings = OllamaEmbeddings()
        
        logging.info('Loading documents from the PDF directory')
        loader = PyPDFDirectoryLoader("./pdfs")
        docs = loader.load()
        
        logging.info('Splitting documents into chunks')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        
        logging.info('Creating a vector store from the documents')
        with ThreadPoolExecutor() as executor:
            future_vectors = executor.submit(FAISS.from_documents, final_documents, embeddings)
            vectors = future_vectors.result()
        
        with open(embeddings_path, 'wb') as f:
            pickle.dump(vectors, f)
    return vectors

# User input for question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to create document embeddings
if st.button("Documents Embedding"):
    with st.spinner('Creating vector store, please wait...'):
        vectors = vector_embedding()
    st.write("Vector Store DB Is Ready")

# If a question is provided, perform the retrieval and display the results
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    vectors = vector_embedding()
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    with st.spinner('Retrieving and processing your question...'):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed_time = time.process_time() - start
        st.write("Response time:", elapsed_time)
        st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
        # Display the relevant chunks
        for i, doc in enumerate(response.get("context", [])):  # Ensure 'context' contains the relevant documents
            st.write(f"Document {i+1}:")
            st.write(doc.page_content)  # Adjust according to the actual structure of your documents
            st.write("--------------------------------")

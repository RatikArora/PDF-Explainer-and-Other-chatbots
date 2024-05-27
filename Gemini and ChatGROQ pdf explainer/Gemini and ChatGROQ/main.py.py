import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up the sidebar
st.sidebar.title("Gemma Model Document Q&A")
st.sidebar.markdown("This application allows you to ask questions based on a set of documents. "
                    "It uses GROQ and Google Generative AI embeddings for document retrieval and question answering.")
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

if page == "Home":
    st.title("Gemma Model Document Q&A")

    # Initialize the language model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    def vector_embedding():
        try:
            if "vectors" not in st.session_state:
                with st.spinner("Loading and processing documents..."):
                    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    st.session_state.loader = PyPDFDirectoryLoader("./d")  # Data Ingestion
                    st.session_state.docs = st.session_state.loader.load()  # Document Loading
                    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)  # Chunk Creation
                    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
                    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
                st.success("Vector Store DB is ready!")
        except Exception as e:
            st.error(f"An error occurred during vector embedding: {e}")

    # User input for the question
    prompt1 = st.text_input("Enter Your Question From Documents")

    # Button to process document embeddings
    if st.button("Create Document Embeddings"):
        vector_embedding()

    # If a question is provided
    if prompt1:
        try:
            with st.spinner("Retrieving the answer..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                start = time.process_time()
                response = retrieval_chain.invoke({'input': prompt1})
                st.write(f"Response time: {time.process_time() - start:.2f} seconds")
                st.write(response['answer'])

                # Display the document similarity search results
                with st.expander("Document Similarity Search"):
                    st.subheader("Relevant Document Chunks:")
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Document {i + 1}:**")
                        st.write(doc.page_content)
                        st.write("--------------------------------")
        except Exception as e:
            st.error(f"An error occurred while processing the question: {e}")

    # Apply some custom styling for better UI

elif page == "About":
    st.title("About This Application")
    st.markdown("""
        This application is designed to provide answers to questions based on a set of uploaded documents. 
        It leverages the power of GROQ and Google Generative AI embeddings to retrieve and process the most relevant information. 
        The key features include:
        - Document ingestion and processing.
        - Vector embeddings for efficient information retrieval.
        - Natural language question answering based on the provided context.
        
        **Technologies Used:**
        - **Streamlit:** For building the web interface.
        - **GROQ:** For language model processing.
        - **Google Generative AI:** For creating embeddings.
        - **FAISS:** For vector similarity search.
    """)

import os
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
class ResumeProcessor:
    def __init__(self, pdf_path: str, embedding_model_name: str = "mistralai/Mistral-7B-v0.1"):
        """
        Initialize the resume processor.
        
        Args:
            pdf_path: Path to the PDF resume file
            embedding_model_name: Name of the HuggingFace embedding model to use
        """
        self.pdf_path = pdf_path
        self.embedding_model_name = embedding_model_name
        self.sections = {
            "education": [],
            "experience": [],
            "projects": [],
            "skills": [],
            "contact": [],
            "awards": []
        }
        self.vectorstore = None
        
    def load_and_process(self):
        """Load PDF, split into sections, and create vector embeddings"""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"Resume PDF not found at {self.pdf_path}")
            
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"Documents: {documents}")
        
        # Split text into manageable chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=350
        )
        
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vectorstore = FAISS.from_documents(texts, embedding_model)
        
        return self.vectorstore
    
    def load_and_process_text(self):
        DATA_FILE = "resume.txt"
        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

        loader = TextLoader(DATA_FILE)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=350)
        texts = text_splitter.split_documents(documents)

        # 3. Create embeddings and a vector store
        embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = FAISS.from_documents(texts, embedding_model)

        # 4. Set up the retriever
        retriever = vectorstore.as_retriever()
        return retriever
    
    def get_retriever(self, search_kwargs=None):
        """Get a retriever from the vectorstore"""
        if self.vectorstore is None:
            self.load_and_process()
            
        if search_kwargs is None:
            search_kwargs = {"k": 3}
            
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
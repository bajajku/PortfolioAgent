from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_and_process_text():
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
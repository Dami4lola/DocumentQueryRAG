import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_to_pinecone(file_path):
    # 1. Load
    print(f"Loading {file_path}...")
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()

    # 2. Split
    print(f"Splitting {len(raw_documents)} pages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Created {len(documents)} text chunks.")

    # 3. Embed & Store
    print("Uploading to Pinecone (this may take a few seconds)...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # This single line handles embedding AND uploading to the database
    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    
    print("SUCCESS! Documents are now stored in the cloud.")

if __name__ == "__main__":
    ingest_to_pinecone("sample.pdf")
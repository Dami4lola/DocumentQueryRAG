import streamlit as st
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Doc Query", page_icon="📄")
st.title("📄 Smart Document Assistant")

# --- SESSION STATE INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(time.time()) # Unique ID for this user session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# --- SIDEBAR: SETTINGS & UPLOAD ---
with st.sidebar:
    st.header("1. Configuration")
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY") 
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if openai_key and pinecone_key:
        st.success("API Keys Loaded")
        client = OpenAI(api_key=openai_key)
        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)
    else:
        st.error("Missing API Keys in .env")
        st.stop()

    st.divider()
    
    st.header("2. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- HELPER FUNCTIONS (Raw Python - No LangChain) ---

def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Simple chunking function to break text into smaller pieces."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap # Move forward but keep some overlap
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def process_and_upload(file, namespace):
    """Reads PDF, chunks it, embeds it, and saves to Pinecone."""
    status = st.empty()
    status.text("Reading PDF...")
    raw_text = get_pdf_text(file)
    
    status.text("Chunking text...")
    text_chunks = chunk_text(raw_text)
    
    status.text(f"Embedding {len(text_chunks)} chunks... (This may take a moment)")
    
    vectors = []
    for i, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk)
        # Create a unique ID for each chunk
        vector_id = f"{namespace}_{i}"
        
        # Prepare vector for Pinecone
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })
    
    # Batch upsert to Pinecone (Max 100 at a time is safer)
    status.text("Saving to Database...")
    index.upsert(vectors=vectors, namespace=namespace)
    
    status.success("Document processed! You can now chat.")
    time.sleep(2)
    status.empty()

# --- MAIN LOGIC ---

# 1. Handle File Upload
if uploaded_file and not st.session_state.file_processed:
    process_and_upload(uploaded_file, st.session_state.session_id)
    st.session_state.file_processed = True

# 2. Chat Interface
if st.session_state.file_processed:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask about your PDF..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant Logic
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                # A. Embed Question
                query_vector = get_embedding(prompt)
                
                # B. Search Pinecone (Only inside this user's namespace)
                search_results = index.query(
                    namespace=st.session_state.session_id, # <--- CRITICAL
                    vector=query_vector,
                    top_k=3,
                    include_metadata=True
                )
                
                matches = [match['metadata']['text'] for match in search_results['matches']]
                context_block = "\n---\n".join(matches)

                # C. Generate Answer
                full_prompt = f"""
                You are a helpful assistant. Answer based ONLY on the context below.
                
                CONTEXT:
                {context_block}
                
                QUESTION:
                {prompt}
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0
                )
                answer = response.choices[0].message.content
                
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                message_placeholder.error(f"Error: {e}")
else:
    st.info("👈 Please upload a PDF file to begin.")

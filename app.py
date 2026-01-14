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
    st.session_state.session_id = str(time.time())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# --- SIDEBAR: SETTINGS & KEYS ---
with st.sidebar:
    st.header("🔐 API Settings")
    
    # Allow user to input their own key
    user_openai_key = st.text_input("Enter YOUR OpenAI Key (for unlimited use)", type="password")
    
    st.divider()
    
    # DECISION LOGIC: Which Key to use?
    if user_openai_key:
        api_key = user_openai_key
        usage_mode = "Unlimited"
        st.success("✅ Using YOUR API Key")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        usage_mode = "Limited"
        remaining = 2 - st.session_state.question_count
        if remaining > 0:
            st.info(f"ℹ️ Trial Mode: {remaining} free questions left.")
        else:
            st.warning("⚠️ Trial Limit Reached.")

    # Always use YOUR Pinecone (It's cheap/safe to share this index for storage)
    pinecone_key = os.getenv("PINECONE_API_KEY") 
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    # Initialize Clients
    if api_key and pinecone_key:
        try:
            client = OpenAI(api_key=api_key)
            pc = Pinecone(api_key=pinecone_key)
            index = pc.Index(index_name)
        except Exception as e:
            st.error(f"Invalid Keys: {e}")
            st.stop()
    else:
        st.error("Missing System API Keys in .env")
        st.stop()

    st.divider()
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- HELPER FUNCTIONS ---

def get_pdf_text(pdf_file, max_pages=5):
    """
    Extract text from PDF.
    SAFETY LIMIT: If using Trial Mode, only read first 5 pages.
    """
    reader = PdfReader(pdf_file)
    text = ""
    
    # Determine page limit based on mode
    num_pages = len(reader.pages)
    if usage_mode == "Limited" and num_pages > max_pages:
        st.warning(f"⚠️ Trial Limit: Only processing the first {max_pages} pages (out of {num_pages}).")
        pages_to_read = reader.pages[:max_pages]
    else:
        pages_to_read = reader.pages

    for page in pages_to_read:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap 
    return chunks

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def process_and_upload(file, namespace):
    status = st.empty()
    status.text("Reading PDF...")
    
    # Pass logic to limit pages if in trial mode
    raw_text = get_pdf_text(file)
    
    status.text("Chunking text...")
    text_chunks = chunk_text(raw_text)
    
    status.text(f"Embedding {len(text_chunks)} chunks...")
    
    vectors = []
    for i, chunk in enumerate(text_chunks):
        embedding = get_embedding(chunk)
        vector_id = f"{namespace}_{i}"
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })
    
    status.text("Saving to Database...")
    index.upsert(vectors=vectors, namespace=namespace)
    
    status.success("Ready! Ask a question below.")
    time.sleep(2)
    status.empty()

# --- MAIN LOGIC ---

# 1. Handle File Upload
if uploaded_file and not st.session_state.file_processed:
    process_and_upload(uploaded_file, st.session_state.session_id)
    st.session_state.file_processed = True

# 2. Chat Interface
if st.session_state.file_processed:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your PDF..."):
        
        # --- BILLING GUARDRAIL ---
        if usage_mode == "Limited" and st.session_state.question_count >= 2:
            st.error("🛑 Free limit reached (2 questions). Please enter your own OpenAI API Key in the sidebar to continue.")
            st.stop() # Halts execution here
        # -------------------------

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                # Increment usage counter if on free tier
                if usage_mode == "Limited":
                    st.session_state.question_count += 1

                # A. Embed Question
                query_vector = get_embedding(prompt)
                
                # B. Search Pinecone
                search_results = index.query(
                    namespace=st.session_state.session_id,
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
                # If their key is bad, tell them
                message_placeholder.error(f"Error: {e}. Check your API Key.")
else:
    st.info("👈 Upload a PDF to start (Max 5 pages in Trial Mode).")

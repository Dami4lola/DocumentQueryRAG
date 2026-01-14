import streamlit as st
import os
from openai import OpenAI
from pinecone import Pinecone

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Doc Query", page_icon="🤖")
st.title("🤖 Smart Document Assistant")

# --- SIDEBAR (Secure Key Entry) ---
with st.sidebar:
    st.header("Settings")
    # Tries to get keys from Secrets first, or asks user to input them
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        openai_key = st.text_input("Enter OpenAI API Key", type="password")
        
    pinecone_key = os.getenv("PINECONE_API_KEY") 
    index_name = os.getenv("PINECONE_INDEX_NAME")

# --- INITIALIZE CLIENTS ---
if openai_key and pinecone_key and index_name:
    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)
else:
    st.warning("⚠️ Please ensure API Keys are set in your .env file or sidebar.")
    st.stop()

# --- CHAT HISTORY (Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CORE RAG LOGIC ---
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def find_best_matches(query_vector):
    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    return [match['metadata']['text'] for match in results['matches']]

# --- USER INPUT ---
if prompt := st.chat_input("Ask a question about your document..."):
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Process (RAG)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # Step A: Embed
            vector = get_embedding(prompt)
            
            # Step B: Search
            matches = find_best_matches(vector)
            context_block = "\n---\n".join(matches)

            # Step C: Prompt
            full_prompt = f"""
            You are a helpful assistant. Answer based ONLY on the context below.
            
            CONTEXT:
            {context_block}
            
            QUESTION:
            {prompt}
            """

            # Step D: Generate
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
            
            # Save Assistant Response
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            message_placeholder.error(f"Error: {e}")
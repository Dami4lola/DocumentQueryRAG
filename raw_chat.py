import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def get_embedding(text):
    """
    Step 1: Turn the user's question into a list of numbers (Vector).
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def find_best_matches(query_vector):
    """
    Step 2: Send those numbers to Pinecone to find similar text chunks.
    """
    results = index.query(
        vector=query_vector,
        top_k=3,           # Return the top 3 best matches
        include_metadata=True # Give us the actual text back, not just IDs
    )
    
    # Extract just the text from the results
    contexts = [match['metadata']['text'] for match in results['matches']]
    return contexts

def generate_answer(query, context_texts):
    """
    Step 3 & 4: Paste the context into a prompt and send to GPT.
    """
    # Join the 3 chunks into one big string
    context_block = "\n---\n".join(context_texts)
    
    prompt = f"""
    You are a helpful assistant. Answer the user's question based ONLY on the context below.
    
    CONTEXT:
    {context_block}
    
    QUESTION:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("--- Raw RAG Chatbot (No LangChain) ---")
    
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        
        print("1. Converting question to vector...")
        vector = get_embedding(user_input)
        
        print("2. Searching Pinecone for matches...")
        matches = find_best_matches(vector)
        
        print(f"   (Found {len(matches)} relevant chunks)")
        
        print("3. Sending to GPT-4...")
        answer = generate_answer(user_input, matches)
        
        print(f"\nAnswer: {answer}")
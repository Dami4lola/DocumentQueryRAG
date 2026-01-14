import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
# These are the "New" imports you already had
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def chat_with_doc():
    print("Initializing ChatBot...")

    # 1. Setup the Brain (The LLM)
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )

    # 2. Setup the Memory (Vector Store)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # 3. Setup the Connection (The NEW Way)
    # We must explicitly define the Prompt Template (Old RetrievalQA did this secretly)
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Create the "Document Chain" (this puts the found docs into the prompt)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the "Retrieval Chain" (this finds the docs first)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    print("\n--- ChatBot Ready! Type 'exit' to quit ---")

    while True:
        query = input("\nAsk a question about your PDF: ")
        
        if query.lower() == "exit":
            break

        # Run the query through the chain
        # Note: The input key is now "input", not just the string
        response = retrieval_chain.invoke({"input": query})
        
        # Note: The answer key is now "answer", not "result"
        print(f"\nAnswer: {response['answer']}")

if __name__ == "__main__":
    chat_with_doc()
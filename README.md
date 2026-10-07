# Papertrail · Document Q&A

A Streamlit demo for chatting with PDF documents. PDFs are split into overlapping passages, embedded with OpenAI, and searched in Pinecone. Answers include the passages they were based on.

## Run locally

1. Install the dependencies:

   ```bash
   pip install -r requirement.txt
   ```

2. Create a `.env` file in the project folder:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=your-pinecone-index-name
   ```

   The Pinecone index must support vectors with 1536 dimensions for `text-embedding-3-small`. You can instead enter an OpenAI API key in the app; Pinecone settings are still required.

3. Start the demo:

   ```bash
   streamlit run app.py
   ```

   Open the local URL printed by Streamlit. Use **Try the sample document** or upload your own PDF from the sidebar.

## Demo notes

- Without a key entered in the app, the configured `OPENAI_API_KEY` is limited to two questions per session and the first five PDF pages.
- Entering a personal OpenAI key in the sidebar enables unlimited questions and removes the five-page limit.
- Each uploaded document is indexed in a separate Pinecone namespace. Replacing it clears the prior namespace.
- PDFs need selectable text; scanned image-only PDFs are not OCR-processed.

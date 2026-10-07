# Papertrail · Document Q&A

A Next.js app for chatting with PDF documents. PDFs are split into overlapping passages, embedded with OpenAI, and searched in Pinecone. Answers stream in and include the passages they were based on. Ready to deploy on Vercel.

## How it works

1. The browser extracts the PDF's text page by page (with [`unpdf`](https://github.com/unjs/unpdf)), so only text is uploaded.
2. `POST /api/ingest` splits the text into 1,000-character passages (800-character stride), embeds them with `text-embedding-3-small`, and stores them in a new Pinecone namespace.
3. `POST /api/ask` embeds the question, retrieves the top 3 passages, and streams an answer from `gpt-4o-mini` that is grounded only in those passages.

Each document gets a signed session token. The token is what lets the browser query (or delete) its own namespace, and it carries the trial question count.

## Run locally

Requires Node.js 20.9+.

1. Install the dependencies:

   ```bash
   npm install
   ```

2. Copy `.env.example` to `.env.local` and fill it in:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=your-pinecone-index-name
   SESSION_SECRET=any-long-random-string
   ```

   The Pinecone index must use 1536 dimensions (for `text-embedding-3-small`). `OPENAI_API_KEY` is optional if every visitor brings their own key; the Pinecone settings are required.

3. Start the dev server and open http://localhost:3000:

   ```bash
   npm run dev
   ```

   Use **Try the sample document** or upload your own PDF.

## Deploy to Vercel

1. Push this repository to GitHub, GitLab or Bitbucket.
2. In Vercel, choose **Add New → Project** and import the repository. The Next.js preset is detected automatically.
3. Add `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` and `SESSION_SECRET` (for example from `openssl rand -hex 32`) under **Environment Variables**.
4. Deploy.

Or, from the command line: `npx vercel`, then `npx vercel env add …` for each variable, then `npx vercel --prod`.

## Demo notes

- Without a personal key, the server's `OPENAI_API_KEY` is limited to two questions and the first five PDF pages.
- Entering a personal OpenAI key in the sidebar removes both limits. The key stays in the browser tab (session storage), is sent only with that visitor's requests, and is never stored on the server.
- Replacing a document deletes the previous one's namespace, and so does removing it or closing the tab.
- The trial limit is enforced with a signed token, not a database, so a determined visitor can reset it by starting over. Add a store such as Upstash Redis if you need a strict limit.
- PDFs need selectable text. Scanned, image-only PDFs are not OCR-processed.
- `public/sample.pdf` is served publicly as the demo document.

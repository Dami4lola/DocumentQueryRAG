import hashlib
import html
import io
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader

load_dotenv()

st.set_page_config(
    page_title="Papertrail · Document Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #202a27;
        --muted: #718078;
        --green: #176b52;
        --green-light: #e9f3ed;
        --line: #e7ebe6;
        --paper: #fbfcf9;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--paper); }
    [data-testid="stSidebar"] {
        background: #f2f5f0;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { font-family: 'Manrope', sans-serif; }
    .block-container { max-width: 1120px; padding-top: 2.25rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -0.035em; }
    .brand {
        color: var(--green); font: 800 1.18rem 'Manrope', sans-serif;
        letter-spacing: -0.04em; margin: 0 0 1.8rem;
    }
    .eyebrow {
        color: var(--green); font-size: .72rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase; margin-bottom: .5rem;
    }
    .hero-title {
        font: 800 clamp(2.2rem, 5vw, 3.65rem)/1.08 'Manrope', sans-serif;
        letter-spacing: -.065em; color: var(--ink); margin: 0;
    }
    .hero-copy { color: var(--muted); font-size: 1.05rem; margin: .8rem 0 1.6rem; }
    .panel {
        background: white; border: 1px solid var(--line); border-radius: 18px;
        padding: 1.25rem 1.4rem; margin: .75rem 0 1.15rem;
        box-shadow: 0 5px 20px rgba(30, 55, 43, .035);
    }
    .doc-label {
        color: var(--muted); font-size: .72rem; font-weight: 700;
        letter-spacing: .1em; text-transform: uppercase;
    }
    .doc-name { font: 700 1.05rem 'Manrope', sans-serif; margin-top: .35rem; }
    .doc-meta { color: var(--muted); font-size: .86rem; margin-top: .25rem; }
    .welcome {
        margin: 2.6rem auto 1.2rem; max-width: 660px; text-align: center;
        padding: 2.35rem 2rem; background: white; border: 1px solid var(--line);
        border-radius: 22px;
    }
    .welcome-icon {
        display: inline-grid; place-items: center; width: 54px; height: 54px;
        background: var(--green-light); border-radius: 17px; font-size: 1.6rem;
        margin-bottom: .8rem;
    }
    .welcome h2 { font-size: 1.45rem; margin: .25rem 0 .45rem; }
    .welcome p { color: var(--muted); margin: 0 auto; max-width: 460px; line-height: 1.6; }
    .section-label {
        font-size: .78rem; font-weight: 700; color: var(--muted);
        text-transform: uppercase; letter-spacing: .09em; margin: 1.8rem 0 .6rem;
    }
    .stButton > button {
        border-radius: 11px; min-height: 2.65rem; font-weight: 600;
        border-color: #dce5dd;
    }
    .stButton > button[kind="primary"] {
        background: var(--green); border-color: var(--green);
    }
    .stChatMessage {
        border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem 1.2rem; background: white; margin: .8rem 0;
    }
    [data-testid="stChatInput"] {
        border-color: #dce5dd; border-radius: 15px;
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid var(--line);
        border-radius: 14px; padding: .85rem 1rem;
    }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "document" not in st.session_state:
    st.session_state.document = None

with st.sidebar:
    st.markdown('<div class="brand">papertrail<span style="color:#a9b8ad">.</span></div>', unsafe_allow_html=True)
    st.markdown("### Your workspace")
    st.caption("Upload a document, then ask questions in plain language.")
    st.divider()

    st.markdown("#### Connect")
    user_openai_key = st.text_input(
        "OpenAI API key",
        type="password",
        placeholder="Paste your key for unlimited questions",
        help="Optional if OPENAI_API_KEY is already configured in your .env file.",
    )
    api_key = user_openai_key or os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    usage_mode = "Unlimited" if user_openai_key else "Trial"
    client = None
    index = None
    connection_error = None
    if not api_key:
        st.warning("Add an OpenAI key above or configure OPENAI_API_KEY in .env.")
    elif not pinecone_key or not index_name:
        missing = "PINECONE_API_KEY" if not pinecone_key else "PINECONE_INDEX_NAME"
        st.warning(f"Configure {missing} in .env to connect your document index.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            index = Pinecone(api_key=pinecone_key).Index(index_name)
            st.success("OpenAI and Pinecone are configured.")
        except Exception as error:
            connection_error = str(error)
            st.error(f"Could not initialize the services: {connection_error}")

    st.divider()
    st.markdown("#### Add a document")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Trial mode processes up to 5 pages.",
    )
    st.caption("PDF only · Up to 5 pages in trial mode")
    if usage_mode == "Trial":
        remaining = max(0, 2 - st.session_state.question_count)
        st.caption(f"Trial questions remaining: **{remaining} / 2**")


st.markdown('<div class="eyebrow">Your documents, understood</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Good questions deserve<br>clear answers.</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Chat with your PDF and get answers grounded in the document.</p>',
    unsafe_allow_html=True,
)

demo_clicked = False
if st.session_state.document is None:
    demo_clicked = st.button("Try the sample document", type="primary", icon="✨")

selected_name = None
selected_bytes = None
if uploaded_file is not None:
    selected_name = uploaded_file.name
    selected_bytes = uploaded_file.getvalue()
elif demo_clicked:
    sample_path = os.path.join(os.path.dirname(__file__), "sample.pdf")
    try:
        with open(sample_path, "rb") as sample_file:
            selected_bytes = sample_file.read()
        selected_name = "sample.pdf"
    except OSError as error:
        st.error(f"Could not load the included sample document: {error}")

if selected_bytes is not None:
    if client is None or index is None:
        st.error("Connect OpenAI and Pinecone in the sidebar before processing a document.")
    else:
        document_hash = hashlib.sha256(selected_bytes).hexdigest()
        current_document = st.session_state.document
        if current_document is None or current_document["hash"] != document_hash:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            try:
                reader = PdfReader(io.BytesIO(selected_bytes))
                page_count = len(reader.pages)
                pages_to_read = reader.pages[:5] if usage_mode == "Trial" else reader.pages
                if usage_mode == "Trial" and page_count > 5:
                    st.info(f"Trial mode will process the first 5 pages of {page_count}.")
                raw_text = "\n".join((page.extract_text() or "") for page in pages_to_read)
                if not raw_text.strip():
                    raise ValueError("No selectable text was found in this PDF.")

                chunks = [
                    raw_text[start : start + 1000]
                    for start in range(0, len(raw_text), 800)
                    if raw_text[start : start + 1000].strip()
                ]
                namespace = str(uuid.uuid4())
                vectors = []
                for chunk_number, chunk in enumerate(chunks):
                    progress_text.caption(f"Preparing your document · {chunk_number + 1} of {len(chunks)}")
                    embedding = client.embeddings.create(
                        input=chunk,
                        model="text-embedding-3-small",
                    ).data[0].embedding
                    vectors.append(
                        {
                            "id": f"{namespace}_{chunk_number}",
                            "values": embedding,
                            "metadata": {"text": chunk},
                        }
                    )
                    progress_bar.progress((chunk_number + 1) / len(chunks))

                index.upsert(vectors=vectors, namespace=namespace)
                old_document = st.session_state.document
                st.session_state.document = {
                    "name": selected_name,
                    "hash": document_hash,
                    "namespace": namespace,
                    "pages": min(page_count, 5) if usage_mode == "Trial" else page_count,
                    "chunks": len(chunks),
                }
                st.session_state.messages = []
                if old_document is not None:
                    try:
                        index.delete(namespace=old_document["namespace"], delete_all=True)
                    except Exception as error:
                        st.warning(f"The previous document could not be removed from the index: {error}")
                progress_text.success("Document ready. Ask away!")
            except Exception as error:
                st.error(f"Could not process this PDF: {error}")
            finally:
                progress_bar.empty()

document = st.session_state.document
if document is None:
    st.markdown(
        """
        <div class="welcome">
            <div class="welcome-icon">📄</div>
            <h2>Start with a PDF</h2>
            <p>Upload a document from the left panel, or try the included sample to see how document Q&amp;A works.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
    step_one, step_two, step_three = st.columns(3)
    with step_one:
        st.markdown("**01 · Add a document**")
        st.caption("Upload a PDF or use the ready-made sample.")
    with step_two:
        st.markdown("**02 · Ask naturally**")
        st.caption("No keywords or special prompts required.")
    with step_three:
        st.markdown("**03 · Get grounded answers**")
        st.caption("See the passages used to answer your question.")
else:
    safe_document_name = html.escape(document["name"])
    st.markdown(
        f"""
        <div class="panel">
            <div class="doc-label">Currently chatting with</div>
            <div class="doc-name">📄 &nbsp;{safe_document_name}</div>
            <div class="doc-meta">{document["pages"]} pages processed &nbsp;·&nbsp; {document["chunks"]} searchable passages</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_questions, metric_chunks, metric_mode = st.columns(3)
    metric_questions.metric("Questions asked", st.session_state.question_count if usage_mode == "Trial" else len(st.session_state.messages) // 2)
    metric_chunks.metric("Searchable passages", document["chunks"])
    metric_mode.metric("Access", "Unlimited" if usage_mode == "Unlimited" else "2-question trial")
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    suggestions = [
        "What is this document about?",
        "Summarize the key points",
        "What should I pay attention to?",
    ]
    if not st.session_state.messages:
        st.caption("Try a question")
        suggestion_columns = st.columns(3)
        for column, suggestion in zip(suggestion_columns, suggestions):
            with column:
                if st.button(suggestion, use_container_width=True):
                    st.session_state.pending_question = suggestion

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Sources · {len(message['sources'])} passages"):
                    for source_number, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**Passage {source_number}**")
                        st.caption(source)

    prompt = st.chat_input("Ask a question about your document...")
    if prompt is None and "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")

    if prompt:
        if client is None or index is None:
            st.error("Connect OpenAI and Pinecone in the sidebar before asking a question.")
        elif usage_mode == "Trial" and st.session_state.question_count >= 2:
            st.error("Your two-question trial is complete. Add your own OpenAI API key in the sidebar to continue.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    if usage_mode == "Trial":
                        st.session_state.question_count += 1
                    query_vector = client.embeddings.create(
                        input=prompt,
                        model="text-embedding-3-small",
                    ).data[0].embedding
                    search_results = index.query(
                        namespace=document["namespace"],
                        vector=query_vector,
                        top_k=3,
                        include_metadata=True,
                    )
                    sources = [
                        match["metadata"]["text"]
                        for match in search_results["matches"]
                        if match.get("metadata") and match["metadata"].get("text")
                    ]
                    if not sources:
                        answer = "I couldn't find a relevant passage in this document to answer that question."
                    else:
                        context_block = "\n---\n".join(sources)
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "Answer the user's question using only the supplied document passages. "
                                        "If the passages do not contain the answer, say so plainly."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        f"DOCUMENT PASSAGES:\n{context_block}"
                                        f"\n\nQUESTION:\n{prompt}"
                                    ),
                                },
                            ],
                            temperature=0,
                        )
                        answer = response.choices[0].message.content
                        if not answer:
                            raise ValueError("The answer service returned an empty response.")
                    st.markdown(answer)
                    if sources:
                        with st.expander(f"Sources · {len(sources)} passages"):
                            for source_number, source in enumerate(sources, start=1):
                                st.markdown(f"**Passage {source_number}**")
                                st.caption(source)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as error:
                    st.error(f"Could not answer this question: {error}")
            st.rerun()

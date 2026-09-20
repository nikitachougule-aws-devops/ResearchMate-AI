import html
import re

import chromadb
import ollama
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ResearchMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# BASIC STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --paper:   #EFEAE0;
        --raised:  #F8F4EA;
        --ink:     #262220;
        --rule:    #C8BFAE;
        --stamp:   #8C2F39;
        --brass:   #A9782F;
        --muted:   #8A8272;
    }

    .stApp { background: var(--paper); }

    html, body, .stMarkdown, p, li, label {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }

    /* Streamlit renders its built-in icons (sidebar arrow, uploader icon, etc.)
       as ligature text in a Material Symbols font. The rule above must not
       touch those spans, or the ligature name prints as literal text. */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
        color: inherit;
    }

    /* ---------- Masthead ---------- */
    .masthead {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        padding: 0.4rem 0 1.1rem;
        border-bottom: 3px double var(--rule);
        margin-bottom: 1.4rem;
    }
    .masthead .stamp {
        flex-shrink: 0;
        width: 62px;
        height: 62px;
        border: 2px solid var(--stamp);
        border-radius: 50%;
        color: var(--stamp);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.7rem;
        transform: rotate(-8deg);
        mix-blend-mode: multiply;
    }
    .masthead h1 {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.5rem;
        line-height: 1;
        margin: 0;
        color: var(--ink);
    }
    .masthead .tagline {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 0.35rem;
        letter-spacing: 0.01em;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--raised);
        border-right: 1px solid var(--rule);
    }
    .brand-lockup {
        display: flex;
        align-items: baseline;
        gap: 0.4rem;
        padding-bottom: 0.9rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--rule);
    }
    .brand-lockup .mark {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-style: italic;
        font-size: 1.35rem;
        color: var(--ink);
    }
    .brand-lockup .tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--stamp);
        border: 1px solid var(--stamp);
        padding: 0.05rem 0.35rem;
        border-radius: 3px;
    }

    section[data-testid="stSidebar"] div.stButton > button {
        background: transparent;
        color: var(--ink);
        border: 1px solid var(--ink);
        border-radius: 2px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: var(--ink);
        color: var(--paper);
    }

    .dev-tag {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 1.4rem;
        padding-top: 0.9rem;
        border-top: 1px dashed var(--rule);
    }
    .dev-tag .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: var(--stamp);
        color: var(--paper);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .dev-tag .label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        color: var(--muted);
        text-transform: lowercase;
    }
    .dev-tag .name {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--ink);
    }

    /* ---------- Main-area elements ---------- */
    div[data-testid="stMetric"] {
        background: var(--raised);
        border: 1px solid var(--rule);
        border-radius: 3px;
        padding: 0.6rem 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--stamp);
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--rule);
        gap: 0.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        color: var(--muted);
    }
    .stTabs [aria-selected="true"] {
        color: var(--stamp) !important;
    }

    .source-pill {
        display: inline-block;
        padding: 0.12rem 0.55rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        border-radius: 2px;
        color: var(--brass);
        background: var(--raised);
        border: 1px solid var(--brass);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
    }

    /* ---------- Flashcards: literal index cards ---------- */
    .flashcard {
        position: relative;
        border: 1px solid var(--rule);
        border-top: 3px dashed var(--rule);
        border-radius: 1px;
        padding: 1rem 1.1rem;
        margin: 0.9rem 0;
        background: var(--raised);
        transform: rotate(-0.35deg);
    }
    .flashcard:nth-child(even) { transform: rotate(0.35deg); }

    .flashcard-question {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--ink);
    }

    .flashcard-answer {
        margin-top: 0.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.86rem;
        color: var(--muted);
        line-height: 1.5;
    }
    .flashcard-answer b {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--stamp);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="masthead">
        <div class="stamp">📚</div>
        <div>
            <h1>ResearchMate AI</h1>
            <div class="tagline">a desk for reading, summarizing, and interrogating your documents</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_llm_text(prompt):
    """Call Ollama and safely support both response formats."""
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )

        if hasattr(response, "message"):
            content = response.message.content
        else:
            content = response["message"]["content"]

        if content is None or not str(content).strip():
            return "The local AI model returned an empty response."

        return str(content).strip()

    except Exception as error:
        st.error(
            f"Could not connect to Ollama using model '{MODEL_NAME}'. "
            "Make sure Ollama is running and the model is installed."
        )
        st.code(str(error))
        return ""


def get_source_pages(metadata):
    result = set()

    for item in metadata or []:
        if item and item.get("page") is not None:
            result.add(item["page"])

    return sorted(result)


def show_source_pages(metadata):
    source_pages = get_source_pages(metadata)

    if source_pages:
        pills = "".join(
            f'<span class="source-pill">📄 Page {page}</span>'
            for page in source_pages
        )
        st.markdown(pills, unsafe_allow_html=True)


def retrieve_context(question, collection, model, source_name, count=5):
    """Retrieve only chunks belonging to the currently uploaded document."""
    collection_count = collection.count()

    if collection_count == 0:
        return "", []

    question_vector = model.encode([question])[0]

    try:
        results = collection.query(
            query_embeddings=[question_vector.tolist()],
            n_results=min(count, collection_count),
            where={"source": source_name},
        )
    except Exception:
        # Fallback for databases created before source metadata was added.
        results = collection.query(
            query_embeddings=[question_vector.tolist()],
            n_results=min(count, collection_count),
        )

    documents = results.get("documents", [[]])[0] or []
    metadata = results.get("metadatas", [[]])[0] or []

    return "\n\n".join(documents), metadata


def parse_flashcards(text):
    """Parse flexible QUESTION/ANSWER output from the local model."""
    pattern = (
        r"QUESTION\s*:\s*(.*?)"
        r"\s*ANSWER\s*:\s*(.*?)"
        r"(?=FLASHCARD\s*\d+|QUESTION\s*:|$)"
    )

    matches = re.findall(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)

    cards = []
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()

        if question and answer:
            cards.append((question, answer))

    return cards


def reset_state():
    keys_to_reset = [
        "processed_file_id",
        "chunks",
        "metadata",
        "number_of_pages",
        "chat_history",
        "summary_result",
        "flashcard_text",
        "agent_result",
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


# =========================================================
# LOAD MODELS AND DATABASE
# =========================================================

model = load_embedding_model()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="research_documents")


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "processed_file_id": None,
    "chunks": [],
    "metadata": [],
    "number_of_pages": 0,
    "chat_history": [],
    "summary_result": None,
    "flashcard_text": None,
    "agent_result": None,
}

for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
            <span class="mark">ResearchMate</span>
            <span class="tag">AI</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a text-based research paper, report, or study document.",
    )

    if st.button("↺ Reset workspace", use_container_width=True):
        reset_state()
        st.rerun()

    st.markdown(
        """
        <div class="dev-tag">
            <div class="avatar">NC</div>
            <div>
                <div class="label">Developed by</div>
                <div class="name">Nikita Chougule</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# REQUIRE A FILE
# =========================================================

if uploaded_file is None:
    st.info("Upload a PDF from the sidebar to begin.")
    st.stop()


# =========================================================
# PROCESS PDF
# =========================================================

file_id = f"{uploaded_file.name}_{uploaded_file.size}"

if st.session_state.processed_file_id != file_id:
    with st.status("Processing document...", expanded=True) as status:
        try:
            status.write("Reading PDF...")
            reader = PdfReader(uploaded_file)
            page_count = len(reader.pages)

            status.write("Splitting text into page-aware chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
            )

            new_chunks = []
            new_metadata = []
            new_ids = []

            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""

                page_text = "".join(
                    character
                    for character in page_text
                    if character.isprintable() or character in "\n\t"
                )

                if not page_text.strip():
                    continue

                page_chunks = splitter.split_text(page_text)

                for chunk_number, chunk in enumerate(page_chunks):
                    new_chunks.append(chunk)
                    new_metadata.append(
                        {
                            "source": uploaded_file.name,
                            "page": page_number,
                            "chunk": chunk_number,
                        }
                    )
                    new_ids.append(
                        f"{uploaded_file.name}_page_{page_number}_chunk_{chunk_number}"
                    )

            if not new_chunks:
                status.update(
                    label="No readable text found in this PDF.",
                    state="error",
                )
                st.error(
                    "This PDF does not contain extractable text. "
                    "Scanned PDFs may require OCR."
                )
                st.stop()

            status.write(f"Creating embeddings for {len(new_chunks)} chunks...")
            vectors = model.encode(new_chunks)

            status.write("Saving vectors...")
            collection.upsert(
                ids=new_ids,
                documents=new_chunks,
                embeddings=vectors.tolist(),
                metadatas=new_metadata,
            )

            st.session_state.processed_file_id = file_id
            st.session_state.chunks = new_chunks
            st.session_state.metadata = new_metadata
            st.session_state.number_of_pages = page_count
            st.session_state.chat_history = []
            st.session_state.summary_result = None
            st.session_state.flashcard_text = None
            st.session_state.agent_result = None

            status.update(
                label="Document ready!",
                state="complete",
                expanded=False,
            )

        except Exception as error:
            status.update(label="Document processing failed.", state="error")
            st.error("An error occurred while processing the PDF.")
            st.exception(error)
            st.stop()


chunks = st.session_state.chunks
metadata = st.session_state.metadata
number_of_pages = st.session_state.number_of_pages

if not chunks:
    st.warning("No readable text is available.")
    st.stop()


# =========================================================
# DOCUMENT INFORMATION
# =========================================================

metric_1, metric_2, metric_3 = st.columns(3)

with metric_1:
    st.metric("Pages", number_of_pages)

with metric_2:
    st.metric("Chunks", len(chunks))

with metric_3:
    st.metric("Stored vectors", collection.count())


# =========================================================
# MAIN TABS
# =========================================================

tab_ask, tab_summary, tab_cards, tab_agent = st.tabs(
    [
        "🔎 Ask",
        "📄 Summarize",
        "🧠 Flashcards",
        "🤖 AI Agent",
    ]
)


# =========================================================
# TAB 1: ASK
# =========================================================

with tab_ask:
    st.subheader("Ask questions")
    st.caption("Answers are generated from the uploaded document.")

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])

            if turn.get("pages"):
                st.caption(
                    "Source pages: "
                    + ", ".join(str(page) for page in turn["pages"])
                )

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        context, retrieved_metadata = retrieve_context(
            question,
            collection,
            model,
            uploaded_file.name,
            count=5,
        )

        if not context.strip():
            answer = "I could not find relevant information in the document."
            source_pages = []
        else:
            prompt = f"""
You are ResearchMate AI, a document-grounded research assistant.

Answer the question using ONLY the supplied context.
Do not use outside knowledge.
Do not invent facts.
If the answer is not present, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

            with st.spinner("ResearchMate is thinking..."):
                answer = get_llm_text(prompt)

            if not answer:
                answer = "No answer was generated."

            source_pages = get_source_pages(retrieved_metadata)

        with st.chat_message("assistant"):
            st.write(answer)

            if source_pages:
                st.caption(
                    "Source pages: "
                    + ", ".join(str(page) for page in source_pages)
                )

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "pages": source_pages,
            }
        )


# =========================================================
# TAB 2: SUMMARY
# =========================================================

with tab_summary:
    st.subheader("Summarize document")
    st.caption("Each section is summarized, then the summaries are combined.")

    if st.button("✨ Generate Summary", key="summary_button"):
        partial_summaries = []
        progress = st.progress(0, text="Starting summary...")

        for index, chunk in enumerate(chunks):
            prompt = f"""
You are ResearchMate AI.

Summarize only the document section below.
Include the main idea, important facts, findings,
methods, and conclusions when available.
Do not invent information.

Document section:
{chunk}

Section summary:
"""

            result = get_llm_text(prompt)

            if result:
                partial_summaries.append(result)

            progress.progress(
                (index + 1) / len(chunks),
                text=f"Summarizing section {index + 1} of {len(chunks)}...",
            )

        progress.empty()

        if partial_summaries:
            combined = "\n\n".join(partial_summaries)

            final_prompt = f"""
You are ResearchMate AI.

Create one clear final summary using ONLY the section
summaries below. Do not add outside information.

Use these headings:
1. Main topic
2. Key points
3. Important findings
4. Methods or technical details
5. Conclusion

Section summaries:
{combined}

Final summary:
"""

            with st.spinner("Combining summaries..."):
                final_summary = get_llm_text(final_prompt)

            st.session_state.summary_result = final_summary

    if st.session_state.summary_result:
        st.markdown("### 📝 Summary")
        st.write(st.session_state.summary_result)

        st.download_button(
            "⬇️ Download summary",
            data=st.session_state.summary_result,
            file_name=f"{uploaded_file.name}_summary.txt",
            mime="text/plain",
            key="download_summary",
        )


# =========================================================
# TAB 3: FLASHCARDS
# =========================================================

with tab_cards:
    st.subheader("Study flashcards")
    st.caption("Generate study questions and reveal their answers.")

    if st.button("🧠 Generate Flashcards", key="flashcard_button"):
        flashcard_context = "\n\n".join(chunks)

        prompt = f"""
You are ResearchMate AI, an educational research assistant.

Create exactly 10 useful study flashcards using ONLY
the document below. Do not use outside knowledge.
Do not invent information. Avoid duplicates.

Use this format for every card:

FLASHCARD 1
QUESTION:
A clear study question
ANSWER:
A short accurate answer

Continue through FLASHCARD 10.

Document:
{flashcard_context}

Flashcards:
"""

        with st.spinner("Creating flashcards..."):
            st.session_state.flashcard_text = get_llm_text(prompt)

    if st.session_state.flashcard_text:
        cards = parse_flashcards(st.session_state.flashcard_text)

        if cards:
            for index, (question_text, answer_text) in enumerate(cards, start=1):
                safe_question = html.escape(question_text)
                safe_answer = html.escape(answer_text)

                st.markdown(
                    f"""
                    <div class="flashcard">
                        <div class="flashcard-question">
                            Card {index}: {safe_question}
                        </div>
                        <div class="flashcard-answer">
                            <b>Answer:</b> {safe_answer}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning(
                "The model returned flashcards in an unexpected format. "
                "The raw response is shown below."
            )
            st.write(st.session_state.flashcard_text)

        st.download_button(
            "⬇️ Download flashcards",
            data=st.session_state.flashcard_text,
            file_name=f"{uploaded_file.name}_flashcards.txt",
            mime="text/plain",
            key="download_flashcards",
        )


# =========================================================
# TAB 4: AI AGENT
# =========================================================

with tab_agent:
    st.subheader("🤖 AI Agent")
    st.caption(
        "Describe a task in natural language. The agent selects a document-based "
        "workflow: question answering, summary, or flashcards."
    )

    agent_request = st.text_area(
        "What should the AI Agent do?",
        placeholder=(
            "Examples:\n"
            "- Explain the main findings in simple English\n"
            "- Summarize the research methods\n"
            "- Create study flashcards about the results"
        ),
        key="agent_request",
    )

    if st.button("🤖 Run AI Agent", key="agent_button"):
        if not agent_request.strip():
            st.warning("Please enter a task for the AI Agent.")
        else:
            classification_prompt = f"""
Classify the user's request into exactly one category:
QUESTION, SUMMARY, or FLASHCARDS.

Return only one category word.

User request:
{agent_request}
"""

            with st.spinner("Agent is planning the task..."):
                intent = get_llm_text(classification_prompt).upper()

            if "FLASHCARD" in intent:
                task_instruction = """
Create 5 study flashcards.
Each card must contain QUESTION and ANSWER.
Use only the supplied context.
"""
            elif "SUMMARY" in intent:
                task_instruction = """
Create a clear document-grounded summary.
Include the main topic, key points, findings,
methods, and conclusion when available.
"""
            else:
                task_instruction = """
Answer the user's request using only the supplied context.
If the answer is unavailable, say:
"I could not find the answer in the document."
"""

            context, retrieved_metadata = retrieve_context(
                agent_request,
                collection,
                model,
                uploaded_file.name,
                count=6,
            )

            if not context.strip():
                st.session_state.agent_result = (
                    "I could not find relevant information in the document."
                )
                agent_pages = []
            else:
                agent_prompt = f"""
You are ResearchMate AI Agent.

Complete the following task:
{task_instruction}

Rules:
- Use only the context.
- Do not use outside knowledge.
- Do not invent information.
- Be clear and concise.

User request:
{agent_request}

Context:
{context}

Result:
"""

                with st.spinner("Agent is completing the task..."):
                    agent_result = get_llm_text(agent_prompt)

                st.session_state.agent_result = (
                    agent_result or "No agent result was generated."
                )
                agent_pages = get_source_pages(retrieved_metadata)

            st.session_state.agent_result_pages = agent_pages

    if st.session_state.agent_result:
        st.markdown("### Agent Result")
        st.write(st.session_state.agent_result)

        agent_pages = st.session_state.get("agent_result_pages", [])

        if agent_pages:
            st.caption(
                "Source pages: "
                + ", ".join(str(page) for page in agent_pages)
            )

        st.download_button(
            "⬇️ Download agent result",
            data=st.session_state.agent_result,
            file_name=f"{uploaded_file.name}_agent_result.txt",
            mime="text/plain",
            key="download_agent_result",
        )
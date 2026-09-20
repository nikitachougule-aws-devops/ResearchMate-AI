# 📚 ResearchMate AI

**ResearchMate AI** is an AI-powered research and document assistant built with Python and Streamlit. It helps users understand PDF documents, ask questions, generate summaries, and create study flashcards.

## 🚀 Features

- 🔎 **Ask Questions:** Ask questions about uploaded PDF documents using semantic search.
- 🤖 **AI Research Agent:** Get document-based explanations, comparisons, study plans, and research assistance.
- 📄 **Document Summarization:** Generate concise summaries from research documents.
- 🧠 **Study Flashcards:** Create interactive question-and-answer flashcards for revision.
- 💾 **Local AI Processing:** Uses Ollama and the Llama 3.2 model for AI-generated responses.
- 🎨 **Interactive Interface:** Built with Streamlit for a simple and user-friendly experience.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web application interface |
| PyPDF | PDF text extraction |
| LangChain Text Splitters | Document chunking |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database and semantic retrieval |
| Ollama | Local LLM execution |
| Llama 3.2 3B | AI response generation |

## 🔄 Project Workflow

1. The user uploads a PDF document.
2. PyPDF extracts text from the document.
3. LangChain divides the extracted text into smaller chunks.
4. Sentence Transformers converts the chunks into embeddings.
5. ChromaDB stores the document embeddings.
6. User questions are converted into embeddings.
7. Relevant document chunks are retrieved using semantic similarity.
8. Ollama uses the retrieved context to generate an answer.
9. The application provides summaries, flashcards, and research assistance.

## 📂 Project Structure

```text
ResearchMate-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── assets/
    └── screenshots/
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ResearchMate-AI.git
cd ResearchMate-AI
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Install Ollama on your computer.

Download the required model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before launching the application.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📄 Usage

1. Launch the application.
2. Upload a text-based PDF from the sidebar.
3. Click **Process PDF**.
4. Use the following features:
   - Ask questions about the document.
   - Interact with the AI Research Agent.
   - Generate a document summary.
   - Create study flashcards.
5. Download the generated summary or flashcards when required.

## 🤖 AI Research Agent

The AI Research Agent provides an agent-style research workflow. It adapts its responses based on the user's request, including:

- Concept explanations
- Document comparisons
- Evidence finding
- Study planning
- Question generation
- Important concept extraction
- Practical applications

The current implementation uses LLM-based orchestration and document retrieval. It is not a fully autonomous tool-calling agent.

## 🗄️ Data Storage

ChromaDB stores document embeddings locally in:

```text
./chroma_db
```

This folder is generated automatically when the application runs and is excluded from Git using `.gitignore`.

## Live app: https://researchmate-ai-hlme7ve5myyyx39pb4ekoj.streamlit.app/

## ⚠️ Limitations

- The application primarily supports text-based PDFs.
- Scanned or image-only PDFs may require OCR.
- Ollama must be installed and running for local AI responses.
- Larger documents may take more time to summarize.
- The current application uses a local LLM setup and requires additional configuration for cloud deployment.

## ☁️ Deployment

The source code can be stored on GitHub.

For deployment on Streamlit Community Cloud, the local Ollama integration must be adapted to use a hosted or remotely accessible LLM service. GitHub Codespaces is not required for uploading the project.

## 🔮 Future Improvements

- OCR support for scanned PDFs
- Multi-document research
- Quiz generation
- Research comparison mode
- Source-level citations
- Export to PDF and DOCX
- Hosted LLM integration
- User authentication
- Advanced tool-calling agent architecture

## 👨‍💻 Author

Nikita Chougule

Developed as an AI-powered research and document assistant project using Python, Streamlit, vector search, and local LLM technology.

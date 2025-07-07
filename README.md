# 🧠 RAG-Based PDF Q&A App with Groq & LLaMA3

This project is a **Retrieval-Augmented Generation (RAG)** powered question-answering system that uses Groq’s blazing-fast **LPU architecture** and the open-source **LLaMA3 8B** model. It enables users to ask questions based on the content of academic research papers (PDFs) and receive context-aware, high-quality answers in real time.

## 🚀 Key Features

- Uploads and processes multiple PDFs from the `research_papers/` directory
- Splits documents into smart chunks using `RecursiveCharacterTextSplitter`
- Generates vector embeddings using `OpenAIEmbeddings`
- Stores documents in a `FAISS` vector database
- Uses `ChatGroq` (LLaMA3-8B-8192) to generate answers based only on relevant document context
- Built with `Streamlit` for a clean and interactive web interface
- Includes document similarity view in an expandable panel

---

## 🛠️ Tech Stack

| Component     | Tool/Library                      |
|---------------|-----------------------------------|
| LLM           | [LLaMA3 8B](w) via [Groq](w)       |
| Embeddings    | [OpenAI Embeddings](w)            |
| Vector DB     | [FAISS](w)                        |
| App Framework | [Streamlit](w)                    |
| LLM Orchestration | [LangChain](w)               |
| PDF Loader    | `PyPDFDirectoryLoader`            |
| Environment   | `dotenv`                          |

---

## 📂 Folder Structure
├── research_papers/ # Your PDF documents
├── app.py # Streamlit app
├── .env # API keys (OPENAI_API_KEY, GROQ_API_KEY)
├── requirements.txt # Python dependencies


---

## ⚙️ How to Run

1. **Clone the repo**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

pip install -r requirements.txt

OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key

streamlit run app.py

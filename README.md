# 🎥 YouTube Video Q&A using LangChain (RAG)

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain** to answer questions from a **YouTube video transcript**.

The system:
1. Loads the transcript of a YouTube video
2. Splits the transcript into chunks
3. Generates embeddings for each chunk
4. Stores them in a **FAISS vector database**
5. Retrieves the most relevant chunks for a question
6. Sends them to an **LLM (OpenAI GPT)** to generate an answer

This allows users to **ask questions about a YouTube video and receive accurate answers based on its transcript.**

---

# 🚀 Features

- Load **YouTube transcripts automatically**
- Chunk long transcripts for better retrieval
- Generate **vector embeddings**
- Store embeddings using **FAISS**
- Retrieve relevant transcript parts using **semantic search**
- Generate answers using **OpenAI GPT**
- Built with **LangChain Runnable pipelines**

---

# 🧠 Architecture (RAG Pipeline)

---

# 📦 Tech Stack

- Python
- LangChain
- OpenAI API
- FAISS Vector Database
- YouTube Transcript Loader
- Dotenv

---

# 📂 Project Structure

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-langchain-rag.git
cd youtube-langchain-rag
```

## 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate

## 3️⃣ Install Dependencies
pip install -r requirements.txt

## Create a .env file in the root directory.
OPENAI_API_KEY=your_openai_api_key

## ▶️ Running the Project
python rag_using_langchaing.py

## Example question used in the code:
Can you summarize the video?
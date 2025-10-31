# 🩺 Medical RAG Chatbot (Groq + LangChain + FAISS + Hugging Face)

A **Retrieval-Augmented Generation (RAG)** based **Medical Chatbot** built using **Groq**, **LangChain**, **FAISS**, and **Hugging Face embeddings**.  
The chatbot answers medical questions based on relevant context retrieved from local documents and can optionally **remember past conversations** using conversational memory.

---

##  Features

✅ **Retrieval-Augmented Generation (RAG)** — Combines LLM intelligence with domain-specific medical data stored in FAISS.  
✅ **Conversational Memory (Optional)** — Maintains chat context for more natural, coherent interactions.  
✅ **Groq-Powered LLMs** — Uses `llama-3.1-8b-instant` for fast, accurate responses.  
✅ **Local Vector Search** — FAISS is used for efficient semantic search on medical text data.  
✅ **Streamlit UI** — Interactive, simple chat interface with message history.  
✅ **Secure API Key Handling** — Uses `.env` file to manage the Groq API key securely.

---

##  Architecture Overview

User Query
│
▼
Chat Interface (Streamlit)
│
▼
Retriever (FAISS + HuggingFace Embeddings)
│
▼
Groq LLM (Llama 3.1)
│
▼
Final Answer (with optional conversation memory)



## Components Used

| Component | Purpose |
|------------|----------|
| **Groq API** | To access high-speed Llama 3.1 model for response generation |
| **LangChain** | To manage RAG pipeline and memory |
| **FAISS** | Vector store for semantic document retrieval |
| **Hugging Face Embeddings** | To convert medical documents into vector embeddings |
| **FLASK** | To build the chatbot interface |
| **Python Dotenv** | For environment variable management |

---

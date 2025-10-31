from flask import Flask, render_template, request, session, jsonify
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# -----------------------------------------------------------
#  Initialize Flask app
# -----------------------------------------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

# -----------------------------------------------------------
#  Load environment and Groq API
# -----------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY in .env file")

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------------------------------------
#  Load FAISS Vector Store
# -----------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

# -----------------------------------------------------------
#  RAG pipeline (with simple memory via Flask session)
# -----------------------------------------------------------
def rag_answer(question):
    # Load past chat memory
    chat_history = session.get("chat_history", [])

    # Build memory context string
    memory_context = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

    # Retrieve relevant documents
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build final prompt
    prompt = f"""
You are a knowledgeable and friendly medical assistant.
Use the conversation history and the retrieved context to answer clearly and accurately.

Conversation History:
{memory_context}

Retrieved Medical Context:
{context}

User Question:
{question}

Helpful and concise answer:
"""

    # Send to Groq model
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    # Save to session memory
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    session["chat_history"] = chat_history

    return answer

# -----------------------------------------------------------
#  Flask routes
# -----------------------------------------------------------
@app.route("/")
def index():
    chat_history = session.get("chat_history", [])
    return render_template("index.html", chat_history=chat_history)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    answer = rag_answer(question)
    return jsonify({"answer": answer})

# -----------------------------------------------------------
#  Run app
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

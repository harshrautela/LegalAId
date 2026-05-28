LegalAId ⚖️🤖

An AI-powered Indian Legal Assistant designed to simplify legal research, document analysis, and legal understanding using Retrieval-Augmented Generation (RAG), Fine-Tuned LLMs, and Vector Search.

🚀 Features
🔍 Retrieval-Augmented Legal Q&A (RAG)
📄 PDF Upload & Legal Document Processing
🧠 Fine-Tuned TinyLlama Legal Model
⚖️ IRAC Argument Drafting
📝 Legal Document Summarization
📚 Grounded Evidence Citation
📊 Comparative AI Model Evaluation
🌙 Premium Dark-Themed Streamlit UI
⚡ FAISS-based High-Speed Vector Search
🛠️ Tech Stack
Backend & AI
Python 3.10
Transformers
PEFT / QLoRA
Sentence Transformers
FAISS
TinyLlama
Groq API
Frontend
Streamlit
Plotly
Altair
Custom CSS
Database & Storage
SQLite
SQLAlchemy
FAISS Vector Index
🧠 System Architecture

User Query → Query Expansion → FAISS Retrieval → Lexical Reranking → Prompt Construction → LLM Inference → Streamlit UI Response

📂 Project Structure
LegalAId/
│
├── data/
├── database/
├── evaluation/
├── models/
├── outputs/
├── prompts/
├── scripts/
├── ui/
├── utils/
├── .streamlit/
├── requirements.txt
├── config.py
├── run.bat
└── README.md
⚡ Main Functionalities
1. Legal Question Answering

Ask legal questions grounded strictly in Indian legal documents and statutes.

2. Document Summarization

Summarizes legal judgments using TF-IDF based extractive summarization.

3. IRAC Legal Drafting

Automatically generates:

Issue
Rule
Application
Conclusion
4. Evidence Grounding

Displays exact legal text chunks used to generate responses.

📊 Performance Highlights
Improved ROUGE-1 score from 42.9% → 63.2%
Reduced hallucination rate from 19.8% → 1.1%
300-word sliding chunk pipeline with 50-word overlap
🔐 Security
API keys secured using .env
Hallucination prevention using grounded prompts and evidence scoring
▶️ Run Locally
# Clone repository
git clone <your-repo-url>

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run ui/app.py
🎯 Target Users
Lawyers
Law Students
Paralegals
Researchers
Citizens seeking legal assistance
📌 Future Improvements
Multi-language legal support
Live Indian law database integration
Voice-based legal assistant
Advanced legal citation engine
👨‍💻 Author

Developed as an AI + LegalTech project focused on grounded and reliable legal assistance for Indian law systems.
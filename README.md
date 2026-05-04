# Medicalguard AI
# 🩺 MedicalGuard AI — Safe Medication Label Analysis with RAG + OCR

A production-style AI system for **medication label understanding and safety-aware information retrieval**, combining **Computer Vision (OCR), LLMs, and Retrieval-Augmented Generation (RAG)**.

⚠️ This system is designed for **information support only** — it does NOT provide medical advice and always encourages consulting a healthcare professional.

---

## 🚀 Overview

MedicalGuard AI processes medication images and extracts relevant drug information using:

- 🧠 **TrOCR (Transformer OCR)** for text extraction
- 🔍 **YOLOv8** for object detection (drug region localization)
- 📚 **RAG pipeline (ChromaDB + embeddings)** for semantic retrieval
- 🤖 **LLM-based reasoning** with strict safety prompts
- 🌐 **FastAPI backend + Streamlit UI**

The system ensures **safe, controlled outputs** by enforcing LLM safety rules.

---

## 🧩 Key Features

- 📷 Upload medication image → detect & extract drug name  
- 🔎 Semantic retrieval from medical knowledge base (ChromaDB)  
- 🧠 LLM reasoning with **safety-aware prompts**  
- ⚠️ Built-in **AI safety guardrails** (no diagnosis or treatment advice)  
- 📊 Risk scoring & explanation output  
- 🧪 Debug mode for system transparency  

---

## 🏗️ System Architecture
Image → YOLOv8 → TrOCR → Extracted Text
↓
RAG Pipeline
(Chunking + Embeddings + ChromaDB)
↓
LLM (Safety Rules)
↓
Final Structured Output


---

## 🛠️ Tech Stack

- **Computer Vision:** YOLOv8, TrOCR  
- **NLP / LLM:** Prompt engineering, safety rules  
- **RAG:** ChromaDB (semantic search)  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **Data:** Drug label datasets, JSON knowledge base  
- **Pipeline:** Chunking (≈350 tokens), overlap (≈50 tokens)

---

## 📊 Example Output

- Drug: **IVABRADINE**  
- Decision: **OK**  
- Risk Score: **0.19**  

**Why?**
- Semantic similarity to known drug information  
- Retrieved knowledge base context  

---

## 🔒 AI Safety Design

This system is built with **strict safety constraints**:

- ❌ No medical diagnosis  
- ❌ No treatment recommendations  
- ❌ No dosage advice  
- ✅ Only provides **factual drug information**  
- ✅ Always suggests consulting a doctor  

Implemented via:
- Prompt-based safety rules  
- Controlled LLM output structure  
- Context-grounded RAG responses  

---

## 📁 Project Structure

## 📁 Project Structure


medicalguard-ai/
│
├── agents/ # Core agent logic
├── api/ # FastAPI backend
├── artifacts/ # Outputs and reports
├── data/ # Knowledge base
├── scripts/ # Chunking, embedding, querying
├── app.py # Streamlit UI
├── requirements.txt
└── README.md


---

## ⚙️ How to Run

```bash
# Clone repo
git clone https://github.com/yourusername/medicalguard-ai.git

# Install dependencies
pip install -r requirements.txt

# Run backend
python api/main.py

# Run UI
streamlit run app.py

Motivation

Medical AI systems must be accurate, explainable, and safe.
This project explores how to combine:

perception (vision)
reasoning (LLM)
grounding (RAG)
and safety (prompt rules)

into a real-world AI pipeline.

📌 Future Improvements
Multi-language support
Better OCR robustness
Larger medical knowledge base
Fine-tuned medical LLM
👩‍💻 Author

Dilrabo Khidirova
ML Engineer | AI Systems | LLMs, RAG, Computer Vision

License

MIT License

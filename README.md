# Code Similarity Analysis Suite

A powerful web application to detect code clones and plagiarism in programming assignments. Built with **Streamlit**.

## ✨ Features

### 1. CodeGNN Clone Detection (C/C++)
- Semantic code similarity detection using **Graph Neural Network**
- Uses **AST + CFG + PDG** (Abstract Syntax Tree + Control Flow Graph + Program Dependence Graph)
- Designed specifically for **C and C++** code
- Powered by a custom trained `CodeGNN` model

### 2. Assignment Code Similarity Checker
- For teachers: Upload multiple student submissions
- Supports **PDF, DOCX, TXT, .py, .java, .cpp**, and more
- Generates **interactive similarity heatmap**
- Shows pairwise similarity scores with detailed results

---

## 📁 Project Structure

```bash
code-similarity-app/
├── models/
│   └── code_gnn_model.pth          # Your trained GNN model (140 KB)
├── utils/
│   ├── __init__.py
│   ├── gnn_inference.py            # CodeGNN loading and inference
│   └── code_extractor.py           # Code extraction from PDF/DOCX
├── app.py                          # Main Streamlit application
├── requirements.txt
└── README.md

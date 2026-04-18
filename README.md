# 🚀 InternNova — AI-Powered Internship Matching Platform

> Find internships that **actually match** your skills using Machine Learning (TF-IDF + Cosine Similarity)

---

## 📁 Project Structure

```
internova/
├── backend/
│   ├── app.py              ← Flask REST API + ML Engine
│   └── requirements.txt    ← Python dependencies
├── frontend/
│   └── index.html          ← Complete React SPA (standalone)
└── README.md
```

---

## ⚡ Quick Start (Frontend Only — No Backend Required)

The frontend works **completely standalone** with a built-in ML fallback engine:

```bash
# Just open the file!
open frontend/index.html
# or
python3 -m http.server 3000   # then visit http://localhost:3000/frontend/
```

That's it. No npm, no node, no installation needed.

---

## 🐍 Full Stack Setup (with Flask Backend)

### 1. Set up Python environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Flask server

```bash
python app.py
# API running at http://localhost:5000
```

### 3. Open the frontend

```bash
open frontend/index.html
```

The frontend auto-detects the backend. If Flask is running, it uses it. Otherwise, it falls back to the client-side ML engine.

---

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Create student/employer account |
| POST | `/auth/login` | JWT authentication |
| POST | `/analyze-resume` | Extract skills + score resume |
| POST | `/find-matches` | TF-IDF cosine similarity matching |
| GET | `/internships` | Browse all internships |
| POST | `/save-internship` | Employer: post new internship |
| POST | `/save-candidate` | Save candidate profile |
| GET | `/candidates` | Employer: view matched candidates |
| GET | `/stats` | Platform statistics |
| GET | `/health` | API health check |

---

## 🤖 ML Architecture

```
Resume Text
    │
    ▼
NLP Preprocessing (tokenization, stopword removal)
    │
    ├─── TF-IDF Vectorization ─────────────── 60% weight
    │    (scikit-learn TfidfVectorizer)
    │    Cosine Similarity vs internship corpus
    │
    └─── Skill Overlap Matching ─────────────40% weight
         (direct skill keyword comparison)
    │
    ▼
Hybrid Score = 0.6 × TF-IDF + 0.4 × Skill Match
    │
    ▼
Ranked Internship Recommendations + Gap Analysis
```

### Why Hybrid?
- **TF-IDF alone** misses explicit skill requirements
- **Keyword matching alone** misses semantic context
- **Hybrid approach** gives the best of both worlds

---

## 🎨 Features

- ✅ **Resume Analysis** — PDF/DOCX/text extraction + NLP
- ✅ **AI Matching** — TF-IDF + Cosine Similarity hybrid
- ✅ **Skill Gap Analyzer** — Shows exact missing skills per internship
- ✅ **Explainable AI** — TF-IDF score + skill score breakdown
- ✅ **Smart Suggestions** — AI-generated resume improvement tips
- ✅ **Employer Module** — Post internships, view matched candidates
- ✅ **Authentication** — JWT-based student/employer login
- ✅ **Full Rankings** — All internships ranked by match %
- ✅ **Demo Mode** — Works without backend (client-side ML)

---

## 📊 Sample Data

8 internship listings included:
1. Frontend Developer Intern — Nexus Technologies
2. Machine Learning Engineer Intern — DataSphere AI
3. Full Stack Developer Intern — CloudBurst Solutions
4. Data Science Intern — AnalyticsIQ
5. Backend Developer Intern — ServerStack Inc
6. Mobile App Developer Intern — AppForge Studio
7. DevOps Engineer Intern — InfraCloud Systems
8. Cybersecurity Intern — SecureNet Labs

---

## 🔌 Adding MongoDB (Production)

Replace the in-memory `DB` dict in `app.py` with:

```python
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["internova"]

# Then replace DB["internships"] with db.internships.find()
```

---

## 🏆 Hackathon Notes

This project demonstrates:
- **Machine Learning** — TF-IDF vectorization, cosine similarity
- **NLP** — Tokenization, stopword removal, entity extraction
- **Full-Stack Development** — React + Flask REST API
- **Real-world Usability** — Auth, roles, employer/student views
- **UI/UX Excellence** — Dark glassmorphism, animations, responsive
- **Explainability** — No black boxes, score breakdowns shown

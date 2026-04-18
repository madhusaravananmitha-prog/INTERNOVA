"""
InternNova Backend - Flask REST API
AI-Powered Internship Matching Engine
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import uuid
import re
import math
from datetime import datetime, timedelta
import hashlib
import secrets

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Text extraction
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── In-Memory Database ────────────────────────────────────────────────────────

DB = {
    "users": {},
    "students": {},
    "employers": {},
    "internships": [],
    "tokens": {}
}

# ─── Sample Internship Dataset ─────────────────────────────────────────────────

SAMPLE_INTERNSHIPS = [
    {
        "id": "int_001",
        "title": "Frontend Developer Intern",
        "company": "Nexus Technologies",
        "location": "Remote",
        "type": "Full-time",
        "duration": "3 months",
        "stipend": "$1500/month",
        "logo": "NT",
        "color": "#6366f1",
        "requiredSkills": ["React", "JavaScript", "HTML", "CSS", "TypeScript", "Git"],
        "description": "We are looking for a frontend developer intern with experience in React and modern JavaScript. You will work on building responsive web applications, implementing UI components, and collaborating with the design team. Strong understanding of HTML, CSS, and TypeScript is required. Experience with Git version control and REST API integration is a plus.",
        "posted": "2024-01-15",
        "deadline": "2024-02-15"
    },
    {
        "id": "int_002",
        "title": "Machine Learning Engineer Intern",
        "company": "DataSphere AI",
        "location": "San Francisco, CA",
        "type": "Full-time",
        "duration": "6 months",
        "stipend": "$2500/month",
        "logo": "DS",
        "color": "#8b5cf6",
        "requiredSkills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "NumPy", "Pandas"],
        "description": "Join our ML team to build and deploy machine learning models at scale. You will work on natural language processing, computer vision, and recommendation systems. Strong Python programming skills and experience with TensorFlow or PyTorch required. Knowledge of statistics, linear algebra, and deep learning architectures is essential.",
        "posted": "2024-01-10",
        "deadline": "2024-02-20"
    },
    {
        "id": "int_003",
        "title": "Full Stack Developer Intern",
        "company": "CloudBurst Solutions",
        "location": "New York, NY",
        "type": "Hybrid",
        "duration": "4 months",
        "stipend": "$2000/month",
        "logo": "CB",
        "color": "#06b6d4",
        "requiredSkills": ["React", "Node.js", "Python", "SQL", "MongoDB", "REST API", "Docker"],
        "description": "Looking for a full stack developer with experience across the entire web stack. You will build features from database to UI, work with microservices architecture, and deploy applications using Docker and Kubernetes. Experience with React frontend, Node.js backend, and SQL/NoSQL databases required.",
        "posted": "2024-01-12",
        "deadline": "2024-02-25"
    },
    {
        "id": "int_004",
        "title": "Data Science Intern",
        "company": "AnalyticsIQ",
        "location": "Austin, TX",
        "type": "Full-time",
        "duration": "3 months",
        "stipend": "$1800/month",
        "logo": "AI",
        "color": "#10b981",
        "requiredSkills": ["Python", "R", "SQL", "Pandas", "NumPy", "Matplotlib", "Tableau", "Statistics"],
        "description": "Join our data science team to analyze large datasets and derive actionable business insights. You will perform exploratory data analysis, build predictive models, create visualizations, and present findings to stakeholders. Proficiency in Python or R, strong SQL skills, and knowledge of statistical methods required.",
        "posted": "2024-01-08",
        "deadline": "2024-02-10"
    },
    {
        "id": "int_005",
        "title": "Backend Developer Intern",
        "company": "ServerStack Inc",
        "location": "Remote",
        "type": "Part-time",
        "duration": "3 months",
        "stipend": "$1200/month",
        "logo": "SS",
        "color": "#f59e0b",
        "requiredSkills": ["Python", "Django", "Flask", "PostgreSQL", "REST API", "Redis", "Linux"],
        "description": "We need a backend developer to help build robust and scalable APIs. You will work with Python frameworks like Django and Flask, design database schemas, implement caching strategies with Redis, and ensure API security. Knowledge of PostgreSQL, Redis, and Linux server management required.",
        "posted": "2024-01-14",
        "deadline": "2024-02-28"
    },
    {
        "id": "int_006",
        "title": "Mobile App Developer Intern",
        "company": "AppForge Studio",
        "location": "Seattle, WA",
        "type": "Full-time",
        "duration": "4 months",
        "stipend": "$2200/month",
        "logo": "AF",
        "color": "#ef4444",
        "requiredSkills": ["React Native", "JavaScript", "iOS", "Android", "Firebase", "Redux", "UI/UX"],
        "description": "Build cross-platform mobile applications using React Native. You will develop features for both iOS and Android, integrate Firebase backend services, manage app state with Redux, and collaborate with UI/UX designers. Experience with React Native, JavaScript, and mobile development best practices required.",
        "posted": "2024-01-11",
        "deadline": "2024-02-18"
    },
    {
        "id": "int_007",
        "title": "DevOps Engineer Intern",
        "company": "InfraCloud Systems",
        "location": "Chicago, IL",
        "type": "Full-time",
        "duration": "6 months",
        "stipend": "$2300/month",
        "logo": "IC",
        "color": "#64748b",
        "requiredSkills": ["Docker", "Kubernetes", "AWS", "CI/CD", "Jenkins", "Terraform", "Linux", "Python"],
        "description": "Join our DevOps team to automate infrastructure provisioning and deployment pipelines. You will work with cloud platforms (AWS, GCP), container orchestration with Kubernetes, implement CI/CD pipelines with Jenkins, and write infrastructure as code with Terraform. Experience with Docker, Kubernetes, and cloud platforms required.",
        "posted": "2024-01-09",
        "deadline": "2024-02-22"
    },
    {
        "id": "int_008",
        "title": "Cybersecurity Intern",
        "company": "SecureNet Labs",
        "location": "Washington, DC",
        "type": "Full-time",
        "duration": "3 months",
        "stipend": "$1900/month",
        "logo": "SN",
        "color": "#dc2626",
        "requiredSkills": ["Network Security", "Penetration Testing", "Python", "Linux", "SIEM", "Cryptography", "Ethical Hacking"],
        "description": "Work on real-world cybersecurity challenges including vulnerability assessment, penetration testing, and security incident response. You will analyze network traffic, conduct security audits, implement security controls, and develop automated security tools. Knowledge of network security principles, ethical hacking, and Python scripting required.",
        "posted": "2024-01-13",
        "deadline": "2024-02-15"
    }
]

# Initialize DB with sample data
DB["internships"] = SAMPLE_INTERNSHIPS

# ─── NLP & ML Utilities ────────────────────────────────────────────────────────

SKILL_KEYWORDS = [
    # Programming Languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin", "r",
    # Frontend
    "react", "vue", "angular", "html", "css", "tailwind", "sass", "bootstrap", "nextjs", "gatsby", "svelte",
    # Backend
    "node.js", "nodejs", "django", "flask", "fastapi", "express", "spring", "laravel", "rails",
    # Mobile
    "react native", "flutter", "ios", "android", "swift", "kotlin",
    # ML/AI
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit-learn", "nlp", "computer vision",
    "numpy", "pandas", "matplotlib", "seaborn", "opencv",
    # Data
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "sqlite",
    "tableau", "power bi", "excel", "data analysis", "data science", "statistics",
    # DevOps/Cloud
    "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "jenkins", "ci/cd", "linux", "git",
    # Other
    "rest api", "graphql", "microservices", "agile", "scrum", "firebase", "redux", "ui/ux",
    "cybersecurity", "networking", "cryptography", "penetration testing",
    "figma", "photoshop", "illustrator"
]

EDUCATION_KEYWORDS = ["bachelor", "master", "phd", "b.tech", "m.tech", "bsc", "msc", "be", "me",
                       "computer science", "information technology", "engineering", "mathematics",
                       "data science", "artificial intelligence", "software"]

STOPWORDS = set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                  "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "shall", "can", "i", "we", "you", "he", "she",
                  "it", "they", "this", "that", "these", "those", "my", "our", "your"])


def preprocess_text(text):
    """Clean and normalize text for NLP processing"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


def extract_skills(text):
    """Extract skills from resume text using keyword matching"""
    text_lower = text.lower()
    found_skills = []

    # Multi-word skills first
    multi_word = [s for s in SKILL_KEYWORDS if ' ' in s]
    for skill in multi_word:
        if skill in text_lower:
            found_skills.append(skill.title())

    # Single word skills
    tokens = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\.]*\b', text_lower))
    for skill in SKILL_KEYWORDS:
        if ' ' not in skill and skill in tokens:
            display = skill.upper() if len(skill) <= 3 else skill.title()
            # Special cases
            if skill in ["python", "javascript", "typescript", "html", "css"]:
                display = skill.title()
            if skill in ["nodejs", "node.js"]:
                display = "Node.js"
            if display not in found_skills:
                found_skills.append(display)

    return list(set(found_skills))[:20]  # Cap at 20 skills


def extract_education(text):
    """Extract education information"""
    lines = text.split('\n')
    education = []
    for line in lines:
        if any(kw in line.lower() for kw in EDUCATION_KEYWORDS):
            line = line.strip()
            if len(line) > 5 and len(line) < 200:
                education.append(line)
    return education[:3]


def extract_contact(text):
    """Extract contact information"""
    contact = {}
    # Email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        contact['email'] = email_match.group()
    # Phone
    phone_match = re.search(r'(\+?[\d\s\-\(\)]{10,15})', text)
    if phone_match:
        contact['phone'] = phone_match.group().strip()
    # LinkedIn
    linkedin_match = re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
    if linkedin_match:
        contact['linkedin'] = linkedin_match.group()
    return contact


def compute_resume_score(skills, education, text):
    """Calculate a resume quality score 0-100"""
    score = 0
    # Skills (40 points)
    skill_score = min(len(skills) * 4, 40)
    score += skill_score
    # Education (20 points)
    edu_score = min(len(education) * 10, 20)
    score += edu_score
    # Length/content (20 points)
    word_count = len(text.split())
    length_score = min(word_count / 25, 20)
    score += length_score
    # Keywords diversity (20 points)
    unique_sections = 0
    for kw in ['experience', 'project', 'education', 'skill', 'achievement', 'certification']:
        if kw in text.lower():
            unique_sections += 1
    score += min(unique_sections * 4, 20)
    return round(min(score, 100))


def hybrid_match(resume_text, internship, resume_skills):
    """
    Hybrid matching: 60% TF-IDF + 40% Skill Overlap
    """
    # TF-IDF cosine similarity (60%)
    internship_text = internship['description'] + ' ' + ' '.join(internship['requiredSkills'])
    
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
        corpus = [preprocess_text(resume_text), preprocess_text(internship_text)]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        tfidf_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        tfidf_score = 0.0

    # Skill overlap (40%)
    resume_skills_lower = set(s.lower() for s in resume_skills)
    required_skills_lower = set(s.lower() for s in internship['requiredSkills'])
    
    if required_skills_lower:
        matched = resume_skills_lower.intersection(required_skills_lower)
        skill_score = len(matched) / len(required_skills_lower)
    else:
        skill_score = 0.0
        matched = set()

    # Hybrid score
    hybrid = (0.6 * tfidf_score) + (0.4 * skill_score)
    match_pct = round(hybrid * 100, 1)

    # Matched and missing skills
    matched_skills = [s for s in internship['requiredSkills'] if s.lower() in resume_skills_lower]
    missing_skills = [s for s in internship['requiredSkills'] if s.lower() not in resume_skills_lower]

    return {
        "match_percentage": match_pct,
        "tfidf_score": round(tfidf_score * 100, 1),
        "skill_score": round(skill_score * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


def generate_suggestions(resume_skills, all_missing_skills, resume_score):
    """Generate AI-based resume improvement suggestions"""
    suggestions = []

    if resume_score < 50:
        suggestions.append({
            "type": "critical",
            "icon": "🚨",
            "title": "Strengthen Your Resume",
            "message": "Your resume score is below average. Add more detailed project descriptions and quantify your achievements."
        })

    if len(resume_skills) < 5:
        suggestions.append({
            "type": "warning",
            "icon": "⚡",
            "title": "Add More Technical Skills",
            "message": "You have fewer than 5 skills listed. Aim for 8-12 relevant technical skills to improve match rates."
        })

    # Most frequently missing skills
    skill_freq = {}
    for skill in all_missing_skills:
        skill_freq[skill] = skill_freq.get(skill, 0) + 1

    top_missing = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_missing:
        skills_str = ', '.join([s[0] for s in top_missing])
        suggestions.append({
            "type": "info",
            "icon": "🎯",
            "title": "High-Demand Skills to Learn",
            "message": f"Skills most often required but missing from your profile: {skills_str}. Learning these would significantly boost your match rate."
        })

    suggestions.append({
        "type": "tip",
        "icon": "💡",
        "title": "Optimize Keyword Density",
        "message": "Mirror language from internship descriptions in your resume. ATS systems rank resumes with matching keywords higher."
    })

    suggestions.append({
        "type": "tip",
        "icon": "📊",
        "title": "Quantify Achievements",
        "message": "Replace vague descriptions with metrics: 'Improved app performance by 40%' beats 'Improved app performance'."
    })

    return suggestions


# ─── Auth Helpers ──────────────────────────────────────────────────────────────

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token():
    return secrets.token_hex(32)

def get_user_from_token(token):
    return DB["tokens"].get(token)

def require_auth(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = get_user_from_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        request.current_user = user
        return f(*args, **kwargs)
    return decorated


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "version": "1.0.0", "name": "InternNova API"})


@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')
    name = data.get('name', '')
    role = data.get('role', 'student')  # student | employer

    if not email or not password or not name:
        return jsonify({"error": "All fields required"}), 400

    if email in DB["users"]:
        return jsonify({"error": "User already exists"}), 409

    user_id = str(uuid.uuid4())
    DB["users"][email] = {
        "id": user_id,
        "email": email,
        "password": hash_password(password),
        "name": name,
        "role": role,
        "created_at": datetime.now().isoformat()
    }

    token = generate_token()
    DB["tokens"][token] = {"user_id": user_id, "email": email, "role": role, "name": name}

    return jsonify({
        "token": token,
        "user": {"id": user_id, "email": email, "name": name, "role": role}
    }), 201


@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').lower().strip()
    password = data.get('password', '')

    user = DB["users"].get(email)
    if not user or user["password"] != hash_password(password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = generate_token()
    DB["tokens"][token] = {
        "user_id": user["id"],
        "email": email,
        "role": user["role"],
        "name": user["name"]
    }

    return jsonify({
        "token": token,
        "user": {"id": user["id"], "email": email, "name": user["name"], "role": user["role"]}
    })


@app.route('/analyze-resume', methods=['POST'])
def analyze_resume():
    """
    Main ML endpoint: analyze resume text, extract info, score it.
    Accepts JSON with 'text' field (resume text) or multipart form with file.
    """
    resume_text = ""

    if request.content_type and 'multipart' in request.content_type:
        file = request.files.get('file')
        if file:
            filename = file.filename.lower()
            file_bytes = file.read()
            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file_bytes)
            elif filename.endswith('.docx'):
                resume_text = extract_text_from_docx(file_bytes)
            else:
                resume_text = file_bytes.decode('utf-8', errors='ignore')
    else:
        data = request.json or {}
        resume_text = data.get('text', '')

    if not resume_text or len(resume_text.strip()) < 20:
        return jsonify({"error": "Could not extract text from resume"}), 400

    # ML Processing
    skills = extract_skills(resume_text)
    education = extract_education(resume_text)
    contact = extract_contact(resume_text)
    score = compute_resume_score(skills, education, resume_text)

    return jsonify({
        "success": True,
        "resume_text": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text,
        "skills": skills,
        "education": education,
        "contact": contact,
        "score": score,
        "word_count": len(resume_text.split()),
        "char_count": len(resume_text)
    })


def extract_text_from_pdf(file_bytes):
    """Extract text from PDF bytes"""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"PDF extraction error: {str(e)}"


def extract_text_from_docx(file_bytes):
    """Extract text from DOCX bytes"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"DOCX extraction error: {str(e)}"


@app.route('/find-matches', methods=['POST'])
def find_matches():
    """
    Core ML matching endpoint.
    Takes resume text + skills, returns ranked internships with match scores.
    """
    data = request.json or {}
    resume_text = data.get('resume_text', '')
    resume_skills = data.get('skills', [])
    top_n = data.get('top_n', 5)

    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400

    # Score all internships
    results = []
    all_missing = []

    for internship in DB["internships"]:
        match_data = hybrid_match(resume_text, internship, resume_skills)
        all_missing.extend(match_data['missing_skills'])
        results.append({
            **internship,
            "match_percentage": match_data["match_percentage"],
            "tfidf_score": match_data["tfidf_score"],
            "skill_score": match_data["skill_score"],
            "matched_skills": match_data["matched_skills"],
            "missing_skills": match_data["missing_skills"]
        })

    # Sort by match percentage
    results.sort(key=lambda x: x["match_percentage"], reverse=True)
    top_results = results[:top_n]

    # Generate suggestions
    suggestions = generate_suggestions(resume_skills, all_missing, 0)

    return jsonify({
        "success": True,
        "total_internships": len(DB["internships"]),
        "matches": top_results,
        "suggestions": suggestions,
        "all_results": results  # For full ranking
    })


@app.route('/save-candidate', methods=['POST'])
def save_candidate():
    data = request.json or {}
    candidate_id = str(uuid.uuid4())
    candidate = {
        "id": candidate_id,
        "created_at": datetime.now().isoformat(),
        **data
    }
    DB["students"][candidate_id] = candidate
    return jsonify({"success": True, "candidate_id": candidate_id})


@app.route('/save-internship', methods=['POST'])
def save_internship():
    data = request.json or {}
    internship_id = f"int_{str(uuid.uuid4())[:8]}"
    internship = {
        "id": internship_id,
        "posted": datetime.now().strftime("%Y-%m-%d"),
        "logo": data.get('company', 'C')[:2].upper(),
        "color": "#6366f1",
        **data
    }
    DB["internships"].append(internship)
    return jsonify({"success": True, "internship": internship}), 201


@app.route('/internships', methods=['GET'])
def get_internships():
    return jsonify({
        "success": True,
        "internships": DB["internships"],
        "total": len(DB["internships"])
    })


@app.route('/candidates', methods=['GET'])
def get_candidates():
    """Employer view - get all candidates with match scores"""
    internship_id = request.args.get('internship_id')
    candidates = list(DB["students"].values())

    if internship_id:
        internship = next((i for i in DB["internships"] if i["id"] == internship_id), None)
        if internship and candidates:
            for candidate in candidates:
                match_data = hybrid_match(
                    candidate.get('resume_text', ''),
                    internship,
                    candidate.get('skills', [])
                )
                candidate['match_percentage'] = match_data['match_percentage']
            candidates.sort(key=lambda x: x.get('match_percentage', 0), reverse=True)

    return jsonify({"success": True, "candidates": candidates})


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_students": len(DB["students"]),
        "total_internships": len(DB["internships"]),
        "total_employers": len(DB["employers"]),
        "avg_match_rate": 72.4  # Mock stat
    })


if __name__ == '__main__':
    print("🚀 InternNova API starting on http://localhost:5000")
    app.run(debug=True, port=5000)

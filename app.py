import streamlit as st
import PyPDF2
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load ML model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Resume Scanner", layout="wide")
st.markdown(
    """
    <style>
        body { background-color: #0b1a2d; color: white; }
        .stTextInput>div>div>input { background-color: #1e2c3a; color: white; }
        .css-18e3th9 { background-color: #0b1a2d; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 AI Resume Scanner")

resumes = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description (Optional)")

def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
    return text

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9., ]+', ' ', text)

def extract_sections(text):
    headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"]
    sections = {h: None for h in headers}
    for header in headers:
        pattern = re.compile(header + r'[:\n][\s\S]{0,500}', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            sections[header] = match.group()
    return sections

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=300, background_color='black', colormap='Blues').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

def resume_match_score(resume, jd):
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([resume, jd])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def smart_suggestions(category):
    suggestions = {
        "Data Scientist": "Add NumPy, Pandas, Deep Learning projects.",
        "Software Engineer": "Include Git, REST APIs, and full-stack apps.",
        "DevOps": "Mention CI/CD, Jenkins, and cloud deployments.",
        "Product Manager": "Describe Agile experience, user stories.",
        "AI Engineer": "Include TensorFlow, NLP projects, or chatbot."
    }
    return suggestions.get(category, "Consider adding more technical and soft skills.")

def detect_strengths(text):
    strengths = []
    keywords = {
        "team": "Teamwork",
        "lead": "Leadership",
        "project": "Project Management",
        "solve": "Problem Solving",
        "design": "Creativity"
    }
    for word, trait in keywords.items():
        if word in text.lower():
            strengths.append(trait)
    return list(set(strengths))

def resume_score(sections, skills_count, match_score):
    section_score = sum([1 for v in sections.values() if v]) / len(sections) * 40
    skill_score = min(skills_count, 10) * 3
    match_score = match_score * 0.3
    total = round(section_score + skill_score + match_score, 2)
    return min(total, 100)

def extract_skills(text):
    keywords = ['python', 'java', 'c++', 'machine learning', 'deep learning', 'sql', 'excel', 
                'communication', 'teamwork', 'problem solving', 'nlp', 'tensorflow', 'keras',
                'pandas', 'numpy', 'scikit-learn', 'html', 'css', 'javascript', 'react', 'django']
    return [kw for kw in keywords if kw in text.lower()]

if resumes:
    for file in resumes:
        st.subheader(f"📄 {file.name}")
        raw = extract_text(file)
        cleaned = clean_text(raw)
        sections = extract_sections(cleaned)
        skills = extract_skills(cleaned)
        strengths = detect_strengths(cleaned)
        job_score = resume_match_score(cleaned, job_desc) if job_desc else 0
        vector_input = vectorizer.transform([cleaned])
        category = model.predict(vector_input)[0]
        total_score = resume_score(sections, len(skills), job_score)

        st.write("### Resume Summary")
        st.write(f"**Predicted Role:** {category}")
        st.write(f"**Match Score with Job Description:** {job_score}%")
        st.write(f"**Overall Resume Score:** {total_score}/100")

        st.write("### Strengths Identified")
        st.write(", ".join(strengths) if strengths else "No major strengths identified.")

        st.write("### Smart Suggestions")
        st.write(smart_suggestions(category))

        st.write("### Extracted Skills")
        st.write(", ".join(skills) if skills else "No specific skills detected.")

        st.write("### Resume Section Check")
        for sec, content in sections.items():
            st.write(f"✅ {sec}" if content else f"❌ {sec}")

        st.write("### Resume Word Cloud")
        generate_wordcloud(cleaned)
else:
    st.info("Upload at least one resume to get started.")

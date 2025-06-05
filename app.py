import streamlit as st
import PyPDF2
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import pickle

model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="AI Resume Scanner", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0b1a2d;
        color: white;
    }
    .stApp {
        background-color: #0b1a2d;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .css-1cpxqw2, .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #1c2c44;
        color: white;
        border-radius: 0.5rem;
    }
    .stButton > button {
        background-color: #0052cc;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1077/1077012.png", width=80)
st.sidebar.title("Resume Scanner")
st.sidebar.markdown("Built with ❤️ using AI")

st.title("🧠 Smart Resume Scanner")
st.markdown("Upload your resume to receive instant AI insights, predictions, and suggestions.")

resumes = st.file_uploader("📤 Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("📝 Paste Job Description (Optional)")

def extract_text(file):
    text = ""
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(file)
    return text

def clean_resume(text):
    return re.sub(r'[^A-Za-z0-9., ]+', ' ', text)

def extract_sections(text):
    sections = {}
    headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"]
    for header in headers:
        pattern = re.compile(header + r'[:\n][\s\S]{0,500}', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            sections[header] = match.group()
    return sections

def resume_match_score(resume, jd):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([resume, jd])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=300, background_color='black', colormap='Blues').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def smart_suggestions(category):
    suggestions = {
        "Data Scientist": "NumPy, Pandas, Deep Learning",
        "Software Engineer": "Git, REST APIs, Docker",
        "DevOps": "Jenkins, CI/CD, Docker",
        "Product Manager": "Agile, Roadmapping, User Stories",
        "AI Engineer": "TensorFlow, PyTorch, NLP"
    }
    return suggestions.get(category, "Keep enhancing your skills!")

if resumes:
    for file in resumes:
        text = extract_text(file)
        cleaned = clean_resume(text)
        st.subheader(f"📄 {file.name}")

        tab1, tab2, tab3 = st.tabs(["📘 Sections", "📊 Match & Prediction", "💡 Suggestions"])

        with tab1:
            st.markdown("### 📌 Extracted Sections")
            sections = extract_sections(cleaned)
            for sec, content in sections.items():
                with st.expander(f"**{sec}**"):
                    st.markdown(content[:1000] + "...")

            st.markdown("### ☁️ Word Cloud of Resume")
            generate_wordcloud(cleaned)

        with tab2:
            if job_desc:
                score = resume_match_score(cleaned, job_desc)
                st.markdown(f"### 📈 Resume-Job Match Score: **{score}%**")
                st.progress(int(score))

            vector_input = vectorizer.transform([cleaned])
            predicted_category = model.predict(vector_input)[0]
            st.markdown(f"### 🔍 Predicted Resume Category: **{predicted_category}**")

        with tab3:
            suggestion = smart_suggestions(predicted_category)
            st.markdown(f"### 🧠 AI Suggestions for You:")
            st.success(f"✅ Add or improve skills like: **{suggestion}**")
else:
    st.info("Upload your resume to get started.")

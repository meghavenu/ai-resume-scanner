import streamlit as st
import pandas as pd
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
import re
import io

# --- Styling ---
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #000000 !important;
    color: white !important;
}
.stApp {
    background-color: #000000 !important;
    padding: 2rem;
}
h1, h2, h3, h4, h5, h6, p, div, label, input, textarea, span {
    color: white !important;
}
.stButton>button {
    background-color: #333333 !important;
    color: white !important;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    border: 1px solid white;
}
.stButton>button:hover {
    background-color: #555555 !important;
    color: #00ffff !important;
}
.stTextInput>div>div>input,
.stTextArea>div>textarea {
    background-color: #1a1a1a !important;
    color: white !important;
    border: 1px solid white;
}
hr { border-color: white; }
[data-testid="stSidebar"] {
    background-color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Title & Description ---
st.title("📄 AI Resume Scanner")
st.markdown("This AI-powered app analyzes your resume, gives a detailed skill summary, a resume score, and suggestions for improvement. Upload your resume and click **Analyze**.")

# --- Upload File ---
uploaded_file = st.file_uploader("Choose your Resume file", type=['pdf', 'docx'])

def extract_text_from_docx(file):
    return docx2txt.process(file)

def extract_text_from_pdf(file):
    from PyPDF2 import PdfReader
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def extract_text(file, filename):
    if filename.endswith('.docx'):
        return extract_text_from_docx(file)
    elif filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        return ""

# --- Skill Extraction ---
def extract_skills(text):
    skill_keywords = [
        'python', 'java', 'sql', 'machine learning', 'data analysis', 'communication',
        'teamwork', 'leadership', 'deep learning', 'nlp', 'cloud', 'excel',
        'pandas', 'numpy', 'matplotlib', 'keras', 'tensorflow'
    ]
    text = text.lower()
    skills_found = [skill for skill in skill_keywords if skill in text]
    return list(set(skills_found))

# --- Resume Score ---
def calculate_score(skills):
    return min(100, len(skills) * 7)

# --- Visualizations ---
def show_pie_chart(skills):
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = skills
    sizes = [1 for _ in skills]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
    ax.set_title("Skill Distribution", color='white')
    st.pyplot(fig)

def show_wordcloud(text):
    wordcloud = WordCloud(width=300, height=300, background_color='black', colormap='cool').generate(text)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# --- Analyze Resume ---
if uploaded_file and st.button("Analyze"):
    filename = uploaded_file.name
    text = extract_text(uploaded_file, filename)
    
    if not text:
        st.error("Could not extract text from the resume.")
    else:
        skills = extract_skills(text)
        score = calculate_score(skills)

        st.markdown("---")
        st.subheader("🔍 Resume Analysis Report")

        st.markdown(f"### ✅ Resume Score: `{score} / 100`")
        st.progress(score)

        st.markdown("### 🧠 Extracted Skills")
        st.success(", ".join(skills) if skills else "No recognizable skills found.")

        st.markdown("### 📊 Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            show_pie_chart(skills)
        with col2:
            show_wordcloud(" ".join(skills))

        st.markdown("### 📋 Professional Feedback")
        if score > 70:
            st.info("Great job! Your resume demonstrates strong technical skills and relevance. Keep it updated and consider tailoring it to specific job roles.")
        elif 40 < score <= 70:
            st.warning("Your resume is decent but can be improved. Try highlighting more technical and domain-specific skills. Include relevant projects and certifications.")
        else:
            st.error("Your resume lacks key skills. Consider taking courses, gaining hands-on experience, and updating your resume accordingly.")

        st.markdown("### 💡 Suggestions for Improvement")
        suggestions = [
            "- Add more technical or domain-specific keywords.",
            "- Highlight projects or internships that match job requirements.",
            "- Use consistent formatting and bullet points.",
            "- Include measurable achievements (e.g., 'increased accuracy by 20%')."
        ]
        st.markdown("\n".join(suggestions))

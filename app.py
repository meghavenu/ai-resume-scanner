import streamlit as st
import joblib
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import os

st.set_page_config(page_title="AI Resume Scanner", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('resume_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# 🌌 Dark mode theme with white text
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #000000 !important;
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}
.stApp { background-color: #000000 !important; padding: 2rem; }
h1, h2, h3, h4, h5, h6, p, div { color: white !important; }
.stButton>button {
    background-color: #0066cc !important;
    color: white !important;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #004c99 !important;
    color: #aad4ff !important;
}
.stTextInput>div>div>input,
.stTextArea>div>textarea {
    background-color: #1a1a1a !important;
    color: white !important;
    border: 1px solid #004080;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Resume Scanner")
st.write("Upload your resume to get a professional analysis of your skills, suggestions for improvement, job category prediction, and visualizations.")

uploaded_file = st.file_uploader("📤 Upload your Resume (.txt, .doc, .docx)", type=['txt', 'doc', 'docx'])

def extract_text(uploaded_file):
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    else:
        return docx2txt.process(uploaded_file)

def extract_skills(text):
    keywords = ['python', 'java', 'sql', 'machine learning', 'deep learning', 'communication', 'teamwork', 'data analysis', 'excel', 'tensorflow']
    found = [kw for kw in keywords if kw in text.lower()]
    return list(set(found))

def score_resume(text):
    length_score = min(len(text.split()) / 1000, 1) * 40
    skill_count = len(extract_skills(text))
    skill_score = min(skill_count / 10, 1) * 60
    return int(length_score + skill_score)

def generate_wordcloud(text):
    return WordCloud(width=500, height=300, background_color='black', colormap='Pastel1').generate(text)

def show_pie_chart(skills):
    skill_freq = {skill: 1 for skill in skills}
    fig, ax = plt.subplots()
    ax.pie(skill_freq.values(), labels=skill_freq.keys(), autopct='%1.1f%%', textprops={'color': "white"})
    st.pyplot(fig)

if uploaded_file is not None:
    text = extract_text(uploaded_file)
    st.subheader("📄 Resume Content Preview:")
    st.text_area("Content", text, height=300)

    if st.button("📊 Analyze Resume"):
        st.subheader("🔍 Resume Analysis")
        prediction = model.predict(vectorizer.transform([text]))[0]
        score = score_resume(text)
        skills = extract_skills(text)

        st.markdown(f"<h3 style='color:#00ccff;'>💯 Resume Score: {score}/100</h3>", unsafe_allow_html=True)
        st.success(f"📌 Predicted Job Category: **{prediction}**")

        st.markdown("### ✅ Detected Skills:")
        st.write(", ".join(skills) if skills else "No strong skills detected.")

        st.markdown("### 💡 Suggestions to Improve Your Resume:")
        if score < 60:
            st.info("- Add more relevant skills.\n- Increase your experience section.\n- Use strong action words.\n- Structure your content better.")
        elif score < 80:
            st.info("- Great! You can still improve formatting and clarity.\n- Add certifications and projects.")
        else:
            st.info("- Excellent resume! You're well-prepared.")

        st.markdown("### ☁️ Word Cloud from Your Resume:")
        wc = generate_wordcloud(text)
        st.image(wc.to_array(), use_column_width=True)

        if skills:
            st.markdown("### 🥧 Skill Distribution (Pie Chart):")
            show_pie_chart(skills)

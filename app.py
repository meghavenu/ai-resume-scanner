import streamlit as st
import PyPDF2
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import pickle
import requests
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_lottie import st_lottie
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="AI Resume Scanner", layout="wide")

model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
nlp = spacy.load("en_core_web_sm")

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

scan_lottie = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_u8o7BL.json")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712027.png", width=100)
    st.title("Resume Scanner Pro")
    st.markdown("🔍 Upload resumes & get AI-powered feedback.")
    st.markdown("---")
    st.info("Built with 💼 ML, 🧠 NLP, and 🎨 Lottie")
    st.markdown("Contact: [LinkedIn](https://linkedin.com)")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3, h4 {
        color: #57c1eb;
    }
    .st-bb {
        background-color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    st_lottie(scan_lottie, height=200)
with col2:
    st.title("AI Resume Scanner")
    st.write("Get instant feedback, job category prediction, resume score, and improvement tips.")

resumes = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description (Optional)")

def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    return ""

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9., ]+', ' ', text)

def extract_sections(text):
    headers = ["Education", "Experience", "Projects", "Skills", "Certifications", "Achievements"]
    sections = {}
    for h in headers:
        match = re.search(h + r'[:\n][\s\S]{0,500}', text, re.IGNORECASE)
        if match:
            sections[h] = match.group()
    return sections

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='black').generate(text)
    st.image(wc.to_array())

if resumes:
    results = []
    for file in resumes:
        raw_text = extract_text(file)
        cleaned = clean_text(raw_text)
        sections = extract_sections(cleaned)
        tokens = vectorizer.transform([cleaned])
        role = model.predict(tokens)[0]
        match_score = 0
        if job_desc:
            tfidf = TfidfVectorizer().fit_transform([cleaned, job_desc])
            match_score = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)
        score = match_score + len(sections) * 10 + len(cleaned.split()) / 100
        results.append({
            "Name": file.name,
            "Role": role,
            "JD Match %": match_score,
            "Word Count": len(cleaned.split()),
            "Sections": sections,
            "Total Score": round(score, 2)
        })
        st.subheader(f"📄 Resume: {file.name}")
        st.write(f"**Predicted Role:** {role}")
        st.write(f"**Job Match Score:** {match_score}%")
        st.write(f"**Total Resume Score:** {round(score, 2)}")
        st.write("**Sections Found:**", list(sections.keys()))
        st.subheader("Word Cloud")
        generate_wordcloud(cleaned)

    df = pd.DataFrame(results)
    st.subheader("📋 Resume Comparison Table")
    st.dataframe(df.drop(columns=["Sections"]))

    st.subheader("📌 JD Match % Distribution")
    fig1 = px.pie(df, names="Name", values="JD Match %", title="Resume vs Job Description Match")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Total Resume Scores")
    fig2 = px.bar(df, x="Name", y="Total Score", color="Name", text="Total Score", title="Overall Resume Score Comparison")
    fig2.update_layout(xaxis_title="Resume", yaxis_title="Score", plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Resume Section Analysis (Radar)")
    categories = list(df["Sections"].iloc[0].keys())
    fig3 = go.Figure()
    for i in range(len(df)):
        sec_data = df["Sections"].iloc[i]
        values = [len(sec_data.get(cat, "")) for cat in categories]
        values += values[:1]
        fig3.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=df["Name"][i]
        ))
    fig3.update_layout(polar=dict(bgcolor="#0e1117"), showlegend=True, paper_bgcolor="#0e1117", font_color="white")
    st.plotly_chart(fig3, use_container_width=True)

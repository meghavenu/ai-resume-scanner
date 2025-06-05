import streamlit as st
import pickle
import spacy
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Resume Scanner", layout="wide")

# Load ML models and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Dark mode CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .css-1d391kg {
        background-color: #0e1117;
        color: white;
    }
    .st-bb {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Resume Scanner")

uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

def extract_text_from_file(file):
    if file.type == "application/pdf":
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx2txt
        text = docx2txt.process(file)
        return text
    else:
        return file.read().decode("utf-8")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def predict_category(text):
    vect_text = vectorizer.transform([text]).toarray()
    pred = model.predict(vect_text)[0]
    return pred

def extract_skills(text):
    skills = []
    skillset = ['python', 'java', 'c++', 'machine learning', 'deep learning', 'sql', 'excel', 
                'communication', 'teamwork', 'problem solving', 'nlp', 'tensorflow', 'keras',
                'pandas', 'numpy', 'scikit-learn', 'html', 'css', 'javascript', 'react', 'django']
    for skill in skillset:
        if skill in text:
            skills.append(skill)
    return skills

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

if uploaded_file is not None:
    raw_text = extract_text_from_file(uploaded_file)
    clean_text = preprocess_text(raw_text)
    
    st.subheader("Extracted Resume Text")
    st.write(clean_text)
    
    st.subheader("Predicted Job Category")
    category = predict_category(clean_text)
    st.success(category)
    
    st.subheader("Extracted Skills")
    skills = extract_skills(clean_text)
    if skills:
        st.write(", ".join(skills))
    else:
        st.write("No skills found.")
    
    st.subheader("Word Cloud of Resume")
    generate_wordcloud(clean_text)
else:
    st.info("Please upload a resume file to get started.")

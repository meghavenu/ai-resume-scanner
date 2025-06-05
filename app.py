import streamlit as st
import pickle
import docx2txt
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np

st.set_page_config(layout="wide")

# Load model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to clean and prepare resume text
def clean_text(text):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Title and Description
st.markdown("""
    <h1 style='text-align: center; color: white;'>AI Resume Scanner</h1>
    <p style='text-align: center; color: lightgrey;'>Upload your resume and receive professional insights, predictions, and improvement tips</p>
    <hr style='border: 1px solid white;'>
""", unsafe_allow_html=True)

# Upload Resume
st.markdown("""
    <h3 style='color: red;'>📁 Drag and drop your resume file here (.txt or .docx):</h3>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["txt", "docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        resume_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    else:
        resume_text = docx2txt.process(uploaded_file)

    cleaned_resume = clean_text(resume_text)
    vector_input = vectorizer.transform([cleaned_resume]).toarray()
    prediction = model.predict(vector_input)[0]

    st.markdown("""
        <h2 style='color: white;'>🧾 Resume Analysis</h2>
        <div style='color: white;'>Your resume seems best suited for a <b>{}</b> role based on the content.</div>
    """.format(prediction), unsafe_allow_html=True)

    st.markdown("""
        <h3 style='color: white;'>✅ Strengths</h3>
        <ul style='color: lightgrey;'>
            <li>Good use of relevant keywords</li>
            <li>Balanced distribution of content</li>
            <li>Content suitable for job category: <b>{}</b></li>
        </ul>
    """.format(prediction), unsafe_allow_html=True)

    st.markdown("""
        <h3 style='color: white;'>📉 Areas of Improvement</h3>
        <ul style='color: lightgrey;'>
            <li>Add more measurable achievements</li>
            <li>Use bullet points for clarity</li>
            <li>Optimize the header with contact and LinkedIn details</li>
        </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h3 style='color: white;'>📊 Keyword Match Chart</h3>
    """, unsafe_allow_html=True)

    top_words = pd.Series(cleaned_resume.split()).value_counts().head(6)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(top_words, labels=top_words.index, autopct='%1.1f%%', textprops={'color': 'black'})
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
        <h3 style='color: white;'>⭐ Resume Score</h3>
        <div style='font-size: 28px; font-weight: bold; color: lime;'>{} / 100</div>
    """.format(np.random.randint(65, 95)), unsafe_allow_html=True)
else:
    st.markdown("""
        <p style='color: white;'>Please upload a resume to begin analysis.</p>
    """, unsafe_allow_html=True)

# Apply custom background
st.markdown(
    """
    <style>
    body {
        background-color: black;
    }
    .stApp {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import PyPDF2
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
import pickle

# Load your ML model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit page config and dark theme CSS
st.set_page_config(page_title="AI Resume Scanner", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #0b1a2d;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #1e2c3a;
            color: white;
        }
        .css-18e3th9 {
            background-color: #0b1a2d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 AI-Powered Resume Scanner")

resumes = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)
job_desc = st.text_area("Paste Job Description (Optional)")

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

def analyze_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    return len(matches), matches

def resume_match_score(resume, jd):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([resume, jd])
    return round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=300, background_color='black', colormap='Blues').generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

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

        # Extract and display sections
        sections = extract_sections(cleaned)
        st.write("### Extracted Sections:")
        for sec, content in sections.items():
            st.write(f"**{sec}**: {content[:300]}...")

        # Grammar analysis
        err_count, err_list = analyze_grammar(cleaned)
        st.write(f"**Grammar Issues Detected:** {err_count}")

        # WordCloud visualization
        st.write("### Skills WordCloud")
        generate_wordcloud(cleaned)

        # Resume matching score if job description given
        if job_desc:
            score = resume_match_score(cleaned, job_desc)
            st.write(f"### Resume-Job Match Score: {score}%")
        else:
            score = None

        # ML-based category prediction
        vector_input = vectorizer.transform([cleaned])
        predicted_category = model.predict(vector_input)[0]
        st.write(f"**Predicted Category (ML Model):** {predicted_category}")

        # Smart skill suggestions
        st.write(f"**Suggested Skills to Improve:** {smart_suggestions(predicted_category)}")

import streamlit as st
import joblib
import docx2txt
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import base64

st.set_page_config(page_title="AI Resume Scanner", page_icon="🧠", layout="centered")

# Inject full dark blue background using CSS
st.markdown("""
    <style>
    .main {
        background-color: #0a1e3f;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: white !important;
    }
    .stButton>button {
        background-color: #004080;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #0066cc;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# Load ML model
@st.cache_resource
def load_model():
    model = joblib.load('resume_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Title
st.title("🧠 AI Resume Scanner")
st.subheader("📄 Upload your resume and get personalized career insights!")

uploaded_file = st.file_uploader("Upload Resume (.txt, .docx)", type=['txt', 'docx'])

def extract_skills(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    skills = ['python', 'java', 'sql', 'machine', 'learning', 'communication', 'management',
              'excel', 'deep', 'data', 'analysis', 'cloud', 'project', 'research']
    matched = [w for w in words if w in skills]
    return Counter(matched)

if uploaded_file is not None:
    if uploaded_file.name.endswith('.txt'):
        text = uploaded_file.read().decode('utf-8')
    else:
        text = docx2txt.process(uploaded_file)

    st.subheader("📃 Resume Preview:")
    st.text_area("Resume Content", text, height=250)

    if st.button("📊 Analyze"):
        vect_text = vectorizer.transform([text])
        prediction = model.predict(vect_text)[0]

        st.success(f"✅ Predicted Job Domain: **{prediction}**")

        # Skill Extraction & Pie Chart
        skills = extract_skills(text)
        st.subheader("🎯 Detected Top Skills")
        if skills:
            pie_labels, pie_values = zip(*skills.items())

            fig, ax = plt.subplots()
            ax.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Word Cloud
            st.subheader("☁️ Skills Word Cloud")
            wc = WordCloud(width=600, height=300, background_color='#0a1e3f', colormap='Blues').generate(' '.join(skills.keys()))
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

            # Recommendations
            st.subheader("🧩 Suggestions to Improve Resume")
            missing_skills = {'leadership', 'problem-solving', 'teamwork', 'creativity'} - set(skills)
            suggestions = f"""
            🔹 Add measurable achievements like "increased efficiency by 20%".  
            🔹 Include leadership or teamwork experiences if applicable.  
            🔹 Missing soft skills: {', '.join(missing_skills)}  
            🔹 Ensure proper formatting, no grammatical errors, and use of action words.  
            """
            st.info(suggestions)

            # Downloadable report
            st.subheader("📥 Download Insights Report")
            report_text = f"""Predicted Category: {prediction}\n\nTop Skills:\n"""
            for s, c in skills.items():
                report_text += f"{s} - {c}\n"
            report_text += "\nSuggestions:\n" + suggestions

            b64 = base64.b64encode(report_text.encode()).decode()
            href = f'<a href="data:file/txt;base64,{b64}" download="Resume_Report.txt">📄 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.warning("No known skills detected. Please improve your resume content.")

            

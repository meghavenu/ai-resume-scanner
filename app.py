import streamlit as st
import joblib
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

st.set_page_config(page_title="AI Resume Scanner", page_icon=":briefcase:", layout="wide")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('resume_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Custom CSS styling for full dark blue background
st.markdown("""
    <style>
    .stApp {
        background-color: #0a1e3f;
        color: white;
    }
    .stButton>button {
        background-color: #004080;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #002050;
        color: #aad4ff;
    }
    .css-1d391kg p {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📝 AI Resume Scanner")
st.write("""
Upload your resume (.txt, .doc, .docx) to analyze your skills and get professional suggestions.
""")

uploaded_file = st.file_uploader("Choose your Resume file", type=['txt', 'doc', 'docx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
        else:
            text = docx2txt.process(uploaded_file)

        st.subheader("📄 Resume Preview:")
        st.text_area("", text, height=250)

        if st.button("📊 Analyze Resume"):
            vect_text = vectorizer.transform([text])
            prediction = model.predict(vect_text)
            st.success(f"**✅ Predicted Job Category:** {prediction[0]}")

            # Extract words for basic skill frequency analysis
            words = re.findall(r"\b\w{4,}\b", text.lower())
            common_skills = ['python', 'java', 'sql', 'machine learning', 'data', 'analysis', 'communication', 'teamwork', 'project']
            found_skills = [word for word in words if word in common_skills]
            skill_freq = Counter(found_skills)

            if found_skills:
                st.subheader("🔍 Skills Frequency (Pie Chart)")
                fig, ax = plt.subplots()
                ax.pie(skill_freq.values(), labels=skill_freq.keys(), autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                st.pyplot(fig)

                st.subheader("☁️ Word Cloud of Resume")
                wc = WordCloud(width=800, height=400, background_color='#0a1e3f', colormap='Blues').generate(text)
                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)

                st.subheader("💡 Professional Feedback")
                st.markdown("""
                - You have strong skills in: **{}**.
                - Consider adding measurable achievements or projects.
                - Make sure your resume has a clean layout with clear sections.
                - Highlight leadership, certifications, or publications if available.
                - Tailor your resume for each job application.
                """.format(', '.join(set(found_skills))))
            else:
                st.warning("⚠️ No common professional skills detected. Consider including specific technical or soft skills.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

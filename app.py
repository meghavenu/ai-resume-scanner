import streamlit as st
import joblib
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import numpy as np
import base64
from io import BytesIO

st.set_page_config(page_title="AI Resume Scanner", page_icon=":briefcase:", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('resume_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

st.markdown("""
<style>
body {
    background-color: #000000 !important;  /* black background */
    color: white !important;              /* white text */
}
.stApp {
    background-color: #000000 !important; /* ensures full app background black */
    color: white !important;
}
.stButton>button {
    background-color: #004080; /* dark blue button */
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
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


st.title("📝 AI Resume Scanner")

st.markdown("""
This application uses machine learning to predict your resume's job category and provide insightful feedback.
Upload your resume file (.txt, .doc, or .docx) and click **Analyze** to get started.
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

            st.markdown(f"<h3 style='color:#aad4ff;'>✅ Predicted Job Category: {prediction[0]}</h3>", unsafe_allow_html=True)

            skills = re.findall(r'\b(?:Python|Java|SQL|Excel|Data Analysis|Machine Learning|Communication|Teamwork|Leadership)\b', text, re.IGNORECASE)
            skills_count = {skill.title(): skills.count(skill) for skill in set(skills)}

            st.markdown("<h4 style='color:#aad4ff;'>🧠 Detected Skills:</h4>", unsafe_allow_html=True)
            if skills_count:
                for skill, count in skills_count.items():
                    st.markdown(f"- **{skill}** ({count} mentions)")
            else:
                st.write("No prominent technical skills detected.")

            st.markdown("<h4 style='color:#aad4ff;'>💬 Feedback:</h4>", unsafe_allow_html=True)
            st.markdown("""
                - Your resume shows strengths in the listed skills.
                - Consider emphasizing measurable achievements.
                - Add keywords related to the job category for better matching.
                - Keep formatting consistent and prioritize relevant experience.
            """)

            st.markdown("<h4 style='color:#aad4ff;'>📈 Resume Score:</h4>", unsafe_allow_html=True)
            resume_score = min(100, len(skills) * 10)
            st.markdown(f"<h2 style='color:lime;'>{resume_score} / 100</h2>", unsafe_allow_html=True)

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<h4 style='color:#aad4ff;'>☁️ Word Cloud:</h4>", unsafe_allow_html=True)
                wordcloud = WordCloud(width=350, height=250, background_color='#0a1e3f', colormap='Blues').generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)

            with col2:
                st.markdown("<h4 style='color:#aad4ff;'>🥧 Skill Distribution:</h4>", unsafe_allow_html=True)
                if skills_count:
                    fig, ax = plt.subplots()
                    ax.pie(skills_count.values(), labels=skills_count.keys(), autopct='%1.1f%%', startangle=140, colors=plt.cm.Blues(np.linspace(0.2, 0.8, len(skills_count))))
                    ax.axis('equal')
                    st.pyplot(fig)
                else:
                    st.write("No skill data available to display.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

import streamlit as st
from io import BytesIO
from collections import Counter
import docx2txt
import PyPDF2
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tempfile

SKILLS = [
    'python', 'java', 'c++', 'sql', 'machine learning', 'deep learning', 'data analysis',
    'communication', 'teamwork', 'leadership', 'project management', 'excel', 'powerpoint',
    'cloud', 'aws', 'azure', 'docker', 'kubernetes', 'react', 'javascript', 'html', 'css',
    'tensorflow', 'pandas', 'numpy', 'git', 'linux', 'problem solving'
]

IMPORTANT_SKILLS = [
    'python', 'machine learning', 'communication', 'project management', 'cloud', 'docker'
]

FEEDBACK = {
    'python': "Great that you know Python, it's a highly sought-after skill.",
    'machine learning': "Including machine learning skills can boost your profile for AI roles.",
    'communication': "Strong communication skills are key for teamwork and leadership.",
    'project management': "Mentioning project management experience adds leadership value.",
    'cloud': "Cloud computing skills like AWS or Azure are highly valuable.",
    'docker': "Knowledge of Docker and containerization improves DevOps fit.",
}

def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(file)
    else:
        return None

def extract_skills(text):
    text_lower = text.lower()
    found_skills = [skill for skill in SKILLS if skill in text_lower]
    return found_skills

def keyword_counts(text):
    words = re.findall(r'\w+', text.lower())
    counts = Counter(words)
    common = counts.most_common(20)
    return common

def generate_wordcloud(text, dark_mode=False):
    bg_color = 'black' if dark_mode else 'white'
    wc = WordCloud(width=600, height=300, background_color=bg_color, colormap='Pastel1').generate(text)
    return wc

def calculate_score(found_skills):
    score = 0
    for skill in IMPORTANT_SKILLS:
        if skill in found_skills:
            score += 1
    return int((score / len(IMPORTANT_SKILLS)) * 100)

def create_summary(text, found_skills, score):
    summary = f"Resume Summary Report\n\n"
    summary += f"Resume Score: {score}%\n\n"
    summary += "Detected Skills:\n"
    summary += ", ".join(found_skills) + "\n\n"
    summary += "Feedback:\n"
    for skill in found_skills:
        if skill in FEEDBACK:
            summary += f"- {skill.title()}: {FEEDBACK[skill]}\n"
    missing = [s for s in IMPORTANT_SKILLS if s not in found_skills]
    if missing:
        summary += "\nConsider adding or highlighting these important skills:\n"
        summary += ", ".join(missing) + "\n"
    summary += "\n--- Extracted Resume Text Preview ---\n"
    summary += text[:2000] + ("..." if len(text) > 2000 else "")
    return summary

st.set_page_config(page_title="Advanced Resume Scanner", layout="wide")

dark_mode = st.sidebar.checkbox("Enable Dark Mode", False)

if dark_mode:
    # Inject some CSS for dark mode background & text
    st.markdown(
        """
        <style>
        .main {
            background-color: #0e1117;
            color: #d7dadc;
        }
        .css-1d391kg {
            background-color: #0e1117;
        }
        </style>
        """, unsafe_allow_html=True)

st.title("Advanced Resume Scanner")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    text = extract_text(uploaded_file)
    if text:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Extracted Resume Text (Preview)")
            st.write(text[:3000] + "..." if len(text) > 3000 else text)

            st.subheader("Keyword Frequency")
            common_words = keyword_counts(text)
            words = [w for w, c in common_words]
            counts = [c for w, c in common_words]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(words[::-1], counts[::-1], color='dodgerblue' if not dark_mode else '#1f77b4')
            ax.set_xlabel("Frequency")
            ax.set_title("Top 20 Keywords")
            st.pyplot(fig)

        with col2:
            st.subheader("Detected Skills")
            found_skills = extract_skills(text)
            if found_skills:
                st.write(", ".join(found_skills))
            else:
                st.write("No predefined skills found.")

            st.subheader("Resume Score")
            score = calculate_score(found_skills)
            st.progress(score / 100)
            st.write(f"Your resume score based on key skills is **{score}%**")

            st.subheader("Feedback and Suggestions")
            for skill in found_skills:
                if skill in FEEDBACK:
                    st.write(f"**{skill.title()}**: {FEEDBACK[skill]}")

            missing_skills = [s for s in IMPORTANT_SKILLS if s not in found_skills]
            if missing_skills:
                st.warning(f"Consider adding or emphasizing these important skills: {', '.join(missing_skills)}")

            st.subheader("Resume Word Cloud")
            wc = generate_wordcloud(text, dark_mode)
            fig_wc, ax_wc = plt.subplots(figsize=(6,3))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            # Download summary report
            summary_text = create_summary(text, found_skills, score)
            st.download_button("Download Resume Summary Report", summary_text, file_name="resume_summary.txt")

    else:
        st.error("Unsupported file type. Please upload PDF or DOCX resumes only.")
else:
    st.info("Please upload a resume file to get started.")

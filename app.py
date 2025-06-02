import streamlit as st
import joblib
import docx2txt
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="AI Resume Scanner", layout="wide", page_icon="🧠")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("resume_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #0a1e3f;
    color: white;
}
h1, h2, h3, h4 {
    color: #aad4ff;
}
.stTextInput, .stFileUploader, .stTextArea, .stButton>button {
    background-color: #004080 !important;
    color: white !important;
    border-radius: 10px !important;
}
.stButton>button:hover {
    background-color: #0059b3 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📂 About")
st.sidebar.info("This AI Resume Scanner predicts the job category of your resume using ML.")
st.sidebar.markdown("Created with 💙 by [Your Name]")

# Main
st.title("🧠 AI Resume Scanner")
uploaded_file = st.file_uploader("Upload Resume (.txt, .docx)", type=["txt", "docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        resume_text = uploaded_file.read().decode("utf-8")
    else:
        resume_text = docx2txt.process(uploaded_file)

    st.subheader("📄 Resume Preview")
    st.text_area("Extracted Text", resume_text, height=300)

    if st.button("📊 Analyze"):
        vect_text = vectorizer.transform([resume_text])
        prediction = model.predict(vect_text)[0]
        proba = model.predict_proba(vect_text)[0]

        st.success(f"**Predicted Category:** {prediction}")

        # Visualization
        st.subheader("📈 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(10, 4))
        class_names = model.classes_
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, proba, color="#aad4ff")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel('Confidence')
        st.pyplot(fig)

        # Download
        st.download_button(
            label="📥 Download Report",
            data=f"Resume Category: {prediction}\n\nFull Confidence Scores:\n" +
                 "\n".join(f"{cls}: {round(score*100, 2)}%" for cls, score in zip(class_names, proba)),
            file_name="resume_report.txt"
        )






import streamlit as st
import joblib
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
import io
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="AI Resume Scanner", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('resume_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Custom CSS
st.markdown("""
    <style>
        body {background-color: #001f3f; color: white;}
        .stApp {background-color: #001f3f;}
        h1, h2, h3, .stMarkdown, .stText, .stTitle {color: #ffffff !important;}
        .stButton>button {
            background-color: #004080; 
            color: white; 
            border-radius: 10px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #002050;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI Resume Scanner")
st.markdown("An advanced resume analyzer that classifies your resume and gives detailed feedback on skills, job role, and improvements. Built using Machine Learning & Streamlit.")

uploaded_file = st.file_uploader("📤 Upload your Resume (.txt, .docx)", type=['txt', 'docx'])

if 'history' not in st.session_state:
    st.session_state.history = []

def generate_feedback(text, prediction):
    feedback = f"Based on the resume, you're best suited for: **{prediction}** role.\n\n"
    feedback += "🧠 **Skills Identified:**\n"

    keywords = ['python', 'machine learning', 'data analysis', 'communication', 'leadership', 'deep learning', 'project management']
    found_skills = [kw for kw in keywords if kw in text.lower()]
    
    if found_skills:
        feedback += ', '.join(found_skills)
    else:
        feedback += "Not enough specific skills found."

    feedback += "\n\n🛠️ **Suggestions to Improve:**\n"
    feedback += "- Include more quantifiable achievements\n"
    feedback += "- Mention specific technologies/tools used\n"
    feedback += "- Highlight your certifications or completed courses\n"
    
    return feedback, found_skills

def plot_pie_chart(skills):
    if not skills:
        skills = ['Other']
    data = {skill: 1 for skill in skills}
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', colors=plt.cm.Blues(range(len(data))))
    return fig

def plot_wordcloud(text):
    wordcloud = WordCloud(width=500, height=300, background_color='black', colormap='Blues').generate(text)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def export_to_pdf(text, prediction, feedback):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Predicted Job Role: {prediction}\n\nResume Text:\n{text[:1000]}...\n\nFeedback:\n{feedback}")
    return pdf.output(dest='S').encode('latin1')

if uploaded_file is not None:
    if uploaded_file.name.endswith('.txt'):
        text = uploaded_file.read().decode('utf-8')
    else:
        text = docx2txt.process(uploaded_file)
        
    st.subheader("📄 Resume Preview")
    st.text_area("", text, height=300)
    
    if st.button("📊 Analyze Resume"):
        vect_text = vectorizer.transform([text])
        prediction = model.predict(vect_text)[0]
        feedback, skills = generate_feedback(text, prediction)
        
        st.success(f"✅ **Predicted Category:** {prediction}")
        st.markdown(f"### 📋 Feedback")
        st.write(feedback)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧩 Skills Pie Chart")
            pie_fig = plot_pie_chart(skills)
            st.pyplot(pie_fig)
        with col2:
            st.markdown("### ☁️ Word Cloud")
            wc_fig = plot_wordcloud(text)
            st.pyplot(wc_fig)

        st.session_state.history.append((prediction, feedback))

        if st.button("📥 Download PDF Report"):
            pdf_bytes = export_to_pdf(text, prediction, feedback)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="resume_report.pdf">Download PDF</a>'
            st.markdown(href, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("### 🕘 Previous Analyses")
        for i, (pred, fb) in enumerate(st.session_state.history[::-1][:3], 1):
            st.markdown(f"**{i}.** {pred}")
            st.markdown(f"_Summary:_ {fb[:150]}...")


        

            

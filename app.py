import streamlit as st
import joblib
import docx2txt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF
import base64
import openai

# Set your OpenAI API key here or use environment variables
openai.api_key = 'YOUR_OPENAI_API_KEY'

st.set_page_config(page_title="AI Resume Scanner", page_icon=":briefcase:", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('resume_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Custom CSS for full dark blue background and styling
st.markdown("""
<style>
body, .main, .block-container {
    background-color: #0a1e3f !important;
    color: white !important;
}
.stButton>button {
    background-color: #004080;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    margin-top: 10px;
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
Upload your resume (.txt, .doc, .docx) to get a detailed analysis including job category prediction, skill insights, word cloud, and personalized feedback.
""")

uploaded_file = st.file_uploader("Choose your Resume file", type=['txt', 'doc', 'docx'])

def calculate_resume_score(text, skills):
    base_score = min(len(text) / 1000, 1) * 50  # length score max 50
    skill_score = min(len(skills) * 10, 50)     # skill score max 50
    return round(base_score + skill_score)

def generate_pdf(report_text, pie_img, wordcloud_img, filename="Resume_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)
    
    pdf.image(pie_img, x=10, y=pdf.get_y()+10, w=90)
    pdf.image(wordcloud_img, x=110, y=pdf.get_y()-60, w=90)
    pdf.output(filename)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def openai_chatbot(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
        else:
            text = docx2txt.process(uploaded_file)
        st.subheader("📄 Resume Preview:")
        st.text_area("", text, height=250)

        if st.button("📊 Analyze Resume"):
            vect_text = vectorizer.transform([text])
            prediction = model.predict(vect_text)[0]
            
            # Simple skill extraction (fake example, replace with your logic)
            common_skills = ['Python', 'Java', 'Machine Learning', 'Data Analysis', 'Communication']
            skills_found = [skill for skill in common_skills if skill.lower() in text.lower()]

            resume_score = calculate_resume_score(text, skills_found)

            st.success(f"**✅ Predicted Job Category:** {prediction}")
            st.write(f"**Resume Score:** {resume_score}/100")
            st.progress(resume_score)

            # Pie chart
            labels = ['Skills Found', 'Skills Missing']
            sizes = [len(skills_found), max(0, 5 - len(skills_found))]
            colors = ['#1f77b4', '#d62728']
            fig1, ax1 = plt.subplots(figsize=(4,4))
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
            ax1.axis('equal')
            st.pyplot(fig1)

            # Word Cloud
            wordcloud = WordCloud(width=400, height=200, background_color='navy', colormap='Pastel1').generate(text)
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.imshow(wordcloud, interpolation='bilinear')
            ax2.axis('off')
            st.pyplot(fig2)

            # Feedback
            st.subheader("📋 Feedback & Suggestions")
            feedback = f"""
            **Strengths:** Your resume contains important skills like {', '.join(skills_found) if skills_found else 'no recognizable skills yet, consider adding relevant ones.'}

            **Areas for Improvement:** Consider expanding on your skills section, highlighting measurable achievements, and tailoring your resume to the target job role.

            **Overall:** Your resume score of {resume_score} indicates {'good' if resume_score > 70 else 'room for improvement'}. Keep refining and updating your resume for best results.
            """
            st.markdown(feedback)

            # PDF download option
            pdf_report_text = f"Job Category Prediction: {prediction}\nResume Score: {resume_score}/100\nSkills Found: {', '.join(skills_found)}\n\nFeedback:\n{feedback}"
            pdf_file_name = "Resume_Report.pdf"
            # Save pie chart and wordcloud images temporarily
            pie_img_path = "pie_chart.png"
            wordcloud_img_path = "wordcloud.png"
            fig1.savefig(pie_img_path)
            fig2.savefig(wordcloud_img_path)
            generate_pdf(pdf_report_text, pie_img_path, wordcloud_img_path, pdf_file_name)
            with open(pdf_file_name, "rb") as pdf_file:
                PDFbyte = pdf_file.read()
            st.download_button(label="📥 Download PDF Report", data=PDFbyte, file_name=pdf_file_name, mime='application/pdf')

            # Chatbot Assistant UI
            st.subheader("🤖 Resume Improvement Assistant")
            user_question = st.text_input("Ask a question about resume writing or improvement:")
            if user_question:
                with st.spinner("Getting advice..."):
                    answer = openai_chatbot(user_question)
                st.write(f"**Assistant:** {answer}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

         

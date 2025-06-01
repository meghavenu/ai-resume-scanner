import streamlit as st
import joblib
import docx2txt

st.set_page_config(page_title="AI Resume Scanner", page_icon=":briefcase:", layout="centered")

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Custom CSS styling for dark blue theme
st.markdown("""
<style>
body {background-color: #0a1e3f; color: white;}
.stButton>button {background-color: #004080; color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px;}
.stButton>button:hover {background-color: #002050; color: #aad4ff;}
.css-1d391kg p {font-size: 16px;}
</style>
""", unsafe_allow_html=True)

st.title("📝 AI Resume Scanner")

st.write("""
Upload your resume (.txt, .doc, .docx) and get the predicted job category using machine learning!
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

    except Exception as e:
        st.error(f"Error processing file: {e}")

        





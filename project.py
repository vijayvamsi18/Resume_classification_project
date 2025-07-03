import pickle
import streamlit as st
import PyPDF2
import docx
from PIL import Image
import base64
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Preprocessing function
def tokenize_docs_with_pos(docs):
    allowed_pos = {'NOUN', 'PROPN', 'VERB', 'ADJ'}
    processed = []

    for doc in nlp.pipe(docs, batch_size=50):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in allowed_pos
            and not token.is_stop
            and not token.is_punct
            and not token.like_num
            and not token.is_space
            and token.is_alpha
        ]
        processed.append(tokens)

    return processed

# Load pipelines and encoder
pipeline1 = pickle.load(open("pipeline1.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Function to extract text from resume
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Set Streamlit page config
st.set_page_config(page_title="Resume Classifier", page_icon="🧠", layout="centered")

# Title and description
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>📄 AI Resume Classifier</h1>
    <p style='text-align: center; color: grey;'>Upload a PDF/DOCX or Paste Resume Text to Predict the Domain</p>
    <hr style='border-top: 1px solid #bbb;'>
    """,
    unsafe_allow_html=True
)

# Input method selection
st.subheader("Choose Input Method")
input_method = st.radio("Select how you want to input the resume:", ("Upload File", "Paste Text"))

text = ""

# File upload
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload your resume here 👇", type=["pdf", "docx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_docx(uploaded_file)

# Paste text
elif input_method == "Paste Text":
    text = st.text_area("Paste your resume text here 👇", height=300)

# Resume preview
if text.strip():
    with st.expander("📄 Resume Preview"):
        st.text_area("Extracted Text", text[:2000], height=300)

    # Predict button
    if st.button("🔍 Predict Domain"):
        with st.spinner("Analyzing your resume..."):
            tokens = tokenize_docs_with_pos([text])[0]
            processed_text = " ".join(tokens)

            pred1 = pipeline1.predict([processed_text])[0]
            decoded1 = label_encoder.inverse_transform([pred1])[0]


        # Styling
        st.markdown(
            """
            <style>
            .pred-card {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                margin: 10px 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display predictions
        st.markdown(f"<div class='pred-card'><strong>Predicted Job Role:</strong> {decoded1}</div>", unsafe_allow_html=True)
        st.success("✅ Prediction complete!")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 13px; color: grey;'>Made with ❤️ using Streamlit & Scikit-Learn</p>
    """,
    unsafe_allow_html=True
)


# 🧠 Resume Classification using NLP and Machine Learning

This project applies Natural Language Processing (NLP) and Machine Learning techniques to classify resumes into categories based on their content. The application is built using Python and deployed with Streamlit for an interactive interface.

## 🚀 Project Highlights

- **Text Preprocessing**: Tokenization, Stopword Removal, POS Tag Filtering using spaCy
- **Feature Extraction**: TF-IDF and Bag of Words
- **Model Building**: Trained 4 models
  - Support Vector Machine (SVM)
  - Random Forest Classifier
  - XGBoost Classifier
  - LightGBM Classifier
- **Model Evaluation**: Accuracy scores compared across TF-IDF and BoW
- **Visualization**: Bar charts to compare model performances
- **Deployment**: Interactive web app using Streamlit

---

## 📁 File Structure

- `project_resume_classification.ipynb` – Main notebook with data processing, model training, evaluation, and deployment preparation
- `models/` – (Optional) Serialized models using `pickle`
- `data/` – (Optional) Dataset of resumes (PDFs or CSVs)
- `app.py` – Streamlit app script (if applicable)
- `requirements.txt` – Python dependencies

---

## 📊 Model Comparison

| Model         | Accuracy (TF-IDF) | Accuracy (BoW) |
|---------------|------------------|----------------|
| SVM           | 1.00             | 0.94           |
| Random Forest | 1.00             | 0.94           |
| XGBoost       | 1.00             | 1.00           |
| LightGBM      | 1.00             | 0.94           |

---

## 🧪 Requirements
```bash
pip install -r requirements.txt
```
**Major Libraries Used:**
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `streamlit`
- `spacy`
- `pandas`
- `matplotlib`
---
## ▶️ Run the App
```bash
streamlit run app.py
```
Or open the `.ipynb` notebook to explore training and evaluation results.
---
## 📌 Author
**Vijay Vamsi**  
Aspiring Data Scientist | Passionate about solving real-world problems with AI and ML  
[LinkedIn Profile](https://www.linkedin.com/in/vijay
-vamsi-ab8139257)
---
## ⭐ GitHub Project Tags
`#NLP` `#MachineLearning` `#ResumeClassification` `#TFIDF` `#BagOfWords` `#Streamlit` `#XGBoost` `#LightGBM`

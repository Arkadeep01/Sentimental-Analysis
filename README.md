# 📝 Sentiment Analysis Web App  
**AI-Powered Review Classification**

This end-to-end Machine Learning project classifies text into **Positive** or **Negative** sentiments. It utilizes a hybrid approach, combining modern NLP preprocessing with a high-performance **TF-IDF + Logistic Regression** pipeline for production-ready inference.

---

## 📊 Features
- **Real-time Prediction:** Enter any product review or social media post for instant sentiment analysis.  
- **Clean Web UI:** Built with **Streamlit** to provide an accessible interface for non-technical users.  
- **Robust Preprocessing:** Automated pipeline to handle URLs, user mentions, hashtags, and stopword removal.  
- **Research & Development:** Includes a full Jupyter Notebook documenting the exploration of **BERT embeddings** and the final model training logic.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **UI Framework:** Streamlit  
- **ML Architecture:** Scikit-learn (Logistic Regression), Joblib  
- **NLP Tools:** NLTK, Regex  
- **Environment:** Pandas, NumPy  

---

## 📂 Project Structure
- **`app.py`** → Production script for the Streamlit web application  
- **`BERT_Sentiment_Analysis.ipynb`** → Data exploration, BERT research, and model training  
- **`sentiment_model.pkl`** → Trained Logistic Regression classifier  
- **`vectorizer.pkl`** → Saved TF-IDF vectorizer  
- **`requirements.txt`** → Project dependencies  

---

## 📥 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```

---

## 🔌 Developer Usage

To integrate this model into your own Python code:

```python
import joblib

# Load the trained assets
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Predict sentiment
text = ["The quality of this product is outstanding!"]
vectorized_input = vectorizer.transform(text)
prediction = model.predict(vectorized_input)[0]

print("Positive" if prediction == 1 else "Negative")
```

---

## 👨‍💻 Author
**Arkadeep Baidya**

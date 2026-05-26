
```markdown
# MBTI Personality Predictor 🎭

A Natural Language Processing (NLP) and Machine Learning project that predicts MBTI personality types from text data. The model is trained on a Reddit MBTI dataset and tested on TV show character dialogues from *Friends* and *Breaking Bad*.

---

## 📌 Project Overview

This project analyzes character dialogue and predicts their MBTI personality type using a trained Logistic Regression model with TF-IDF features. 

It also compares predicted results with fan-consensus MBTI types to evaluate performance.

---

## 🎯 Features

- Text preprocessing (lowercasing, stopword removal, lemmatization)
- TF-IDF based feature extraction
- Logistic Regression classifier
- MBTI prediction from text input
- TV character personality analysis
- Streamlit interactive web app

---

## 📊 Datasets Used

- **Training Data:** Kaggle MBTI Dataset (Reddit posts)
- **Test Data:**
  - Friends TV Show transcripts
  - Breaking Bad dialogue dataset

---

## 🧠 Model Pipeline

1. Data Cleaning & Preprocessing
2. Tokenization & Lemmatization (NLTK)
3. TF-IDF Vectorization
4. Logistic Regression Model Training
5. Prediction on character dialogues

---

## 🖥️ Web App

Built using **Streamlit**, the app includes:

- Character MBTI results
- Comparison with fan consensus
- Interactive “Try It Yourself” input
- Visual character display

---

## 📁 Project Structure

```text
MBTI_Predictor/
│
├── data/
│   ├── mbti_1.csv
│   ├── BB_data.csv
│   └── Friends_Transcript.txt
│
├── images/
│
├── source/
│   ├── app.py
│   ├── mbti_pipeline.py
│   └── mbti_predictor.ipynb
│
├── trained model/
│   ├── log_reg_mbti_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── README.md
└── requirements.txt

```

---

## 🚀 How to Run

### 1. Clone repo

```bash
git clone https://github.com/maheen-736/mbti-personality-predictor.git
cd mbti-personality-predictor

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. Run Streamlit app

```bash
streamlit run source/app.py

```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* NLTK
* Pandas
* Streamlit

---

## 📈 Results Summary

* Some characters matched exactly with fan consensus
* Most predictions were within 1 MBTI dimension
* Model performs well on general personality trends

---

## ⚠️ Note

This project is for educational purposes only. MBTI predictions are based on text patterns and not clinical or psychological evaluation.

---

## 📚 References

* [Kaggle MBTI Dataset](https://www.kaggle.com/datasnaek/mbti-type)
* [Scikit-learn Documentation](https://scikit-learn.org/)
* [NLTK Documentation](https://www.nltk.org/)
* [Streamlit Documentation](https://streamlit.io/)

```

```

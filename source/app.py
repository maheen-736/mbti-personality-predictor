import streamlit as st
import pandas as pd
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "../images")

st.set_page_config(page_title="MBTI Predictor", page_icon="🧠", layout="wide")
st.title("🧠 MBTI Personality Predictor")
st.subheader("TV Show Character Analysis — Friends & Breaking Bad")

vectorizer = joblib.load(os.path.join(BASE_DIR, "../trained model/tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(BASE_DIR, "../trained model/log_reg_mbti_model.joblib"))


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


def predict_mbti(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]


character_images = {
    "Ross": "ross.jpg",
    "Rachel": "rachel.jpg",
    "Monica": "monica.jpg",
    "Chandler": "chandler.jpg",
    "Joey": "joey.jpg",
    "Phoebe": "phoebe.jpg",
    "Walter": "walter.jpg",
    "Jesse": "jesse.jpg",
    "Skyler": "skyler.jpg",
    "Hank": "hank.jpg",
    "Walter Jr": "walter_jr.jpg",
}

mbti_descriptions = {
    "INTJ": "Strategic, independent, determined",
    "INTP": "Logical, analytical, curious",
    "ENTJ": "Bold, imaginative, strong-willed",
    "ENTP": "Smart, curious, loves debate",
    "INFJ": "Insightful, principled, compassionate",
    "INFP": "Idealistic, empathetic, creative",
    "ENFJ": "Charismatic, inspiring, empathetic",
    "ENFP": "Enthusiastic, creative, sociable",
    "ISTJ": "Responsible, thorough, dependable",
    "ISFJ": "Supportive, reliable, patient",
    "ESTJ": "Organized, honest, dedicated",
    "ESFJ": "Caring, social, popular",
    "ISTP": "Practical, observant, reserved",
    "ISFP": "Flexible, charming, sensitive",
    "ESTP": "Energetic, perceptive, direct",
    "ESFP": "Spontaneous, energetic, enthusiastic",
}

tab1, tab2, tab3 = st.tabs(["📊 Results", "🎭 Characters", "✍️ Try It Yourself"])

# Tab 1 - Results
with tab1:
    st.header("Predicted vs Fan Consensus MBTI")
    df = pd.read_csv(os.path.join(BASE_DIR, "../data/final_results.csv"))

    for show in df["show"].unique():
        st.subheader(f"📺 {show}")
        show_df = df[df["show"] == show]
        cols = st.columns(len(show_df))
        for i, (_, row) in enumerate(show_df.iterrows()):
            with cols[i]:
                img_file = character_images.get(row["character"])
                img_path = os.path.join(IMAGES_DIR, img_file) if img_file else None
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                st.markdown(f"**{row['character']}**")
                st.markdown(f"🔮 Predicted: `{row['predicted_mbti']}`")
                st.markdown(f"👥 Fan Consensus: `{row['fan_consensus_mbti']}`")
                match = row['predicted_mbti'] == row['fan_consensus_mbti']
                st.markdown("✅ Match!" if match else "❌ No Match")

# Tab 2 - Characters
with tab2:
    st.header("Character MBTI Breakdown")
    df = pd.read_csv(os.path.join(BASE_DIR, "../data/final_results.csv"))
    selected_show = st.selectbox("Select a show:", df["show"].unique())
    show_df = df[df["show"] == selected_show]
    cols = st.columns(len(show_df))
    for i, (_, row) in enumerate(show_df.iterrows()):
        with cols[i]:
            img_file = character_images.get(row["character"])
            img_path = os.path.join(IMAGES_DIR, img_file) if img_file else None
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            st.metric(row["character"], row["predicted_mbti"])
            st.caption(mbti_descriptions.get(row["predicted_mbti"], ""))

# Tab 3 - Try It Yourself
with tab3:
    st.header("✍️ Enter Your Own Text — Get Your MBTI!")
    user_input = st.text_area("Type or paste any text below:", height=150, placeholder="Type something here...")

    if st.button("Predict MBTI 🔮"):
        if user_input.strip():
            result = predict_mbti(user_input)
            st.success(f"Predicted MBTI Type: **{result}**")
            st.info(mbti_descriptions.get(result, ""))

            # Find matching characters from both shows
            df = pd.read_csv(os.path.join(BASE_DIR, "../data/final_results.csv"))
            matched = df[df["predicted_mbti"] == result]

            if not matched.empty:
                st.markdown("### 🎭 Characters with the same MBTI type:")
                cols = st.columns(len(matched))
                for i, (_, row) in enumerate(matched.iterrows()):
                    with cols[i]:
                        img_file = character_images.get(row["character"])
                        img_path = os.path.join(IMAGES_DIR, img_file) if img_file else None
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        st.markdown(f"**{row['character']}**")
                        st.caption(f"📺 {row['show']}")
            else:
                st.markdown("No character in our dataset has this exact MBTI type.")
        else:
            st.warning("Please enter some text first!")
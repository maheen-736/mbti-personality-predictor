import joblib
import pandas as pd
import re
import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Model load karo
vectorizer = joblib.load("../trained model/tfidf_vectorizer.joblib")
model = joblib.load("../trained model/log_reg_mbti_model.joblib")

print("Model loaded successfully!")

# ============ TEXT CLEANING ============
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Punctuation remove
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Stop words remove
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# MBTI predict karo
def predict_mbti(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

# ============ FRIENDS ============
print("\nProcessing Friends...")

with open("../data/friends_data/Friends_Transcript.txt", "r", encoding="latin-1") as f:
    transcript = f.read()

pattern = re.compile(r'^([A-Z][a-zA-Z ]+):\s(.+)$', re.MULTILINE)
matches = pattern.findall(transcript)

dialogue_df = pd.DataFrame(matches, columns=["character", "dialogue"])
main_characters = ["Ross", "Rachel", "Monica", "Chandler", "Joey", "Phoebe"]
dialogue_df = dialogue_df[dialogue_df["character"].isin(main_characters)]

character_dialogues = dialogue_df.groupby("character")["dialogue"].apply(" ".join).reset_index()
character_dialogues["predicted_mbti"] = character_dialogues["character"].apply(
    lambda name: predict_mbti(
        character_dialogues[character_dialogues["character"] == name]["dialogue"].values[0]
    )
)

fan_consensus_friends = {
    "Ross": "ISTJ",
    "Rachel": "ESFP",
    "Monica": "ESTJ",
    "Chandler": "ENTP",
    "Joey": "ESFP",
    "Phoebe": "ENFP"
}
character_dialogues["fan_consensus_mbti"] = character_dialogues["character"].map(fan_consensus_friends)
character_dialogues["show"] = "Friends"
print("Friends done!")

# ============ BREAKING BAD ============
print("\nProcessing Breaking Bad...")

bb_df = pd.read_csv("../data/bb_data/BB_data.csv")
bb_main = ["Walter", "Jesse", "Skyler", "Hank", "Walter Jr"]
bb_df = bb_df[bb_df["actor"].isin(bb_main)]

bb_dialogues = bb_df.groupby("actor")["text"].apply(" ".join).reset_index()
bb_dialogues.columns = ["character", "dialogue"]
bb_dialogues["predicted_mbti"] = bb_dialogues["character"].apply(
    lambda name: predict_mbti(
        bb_dialogues[bb_dialogues["character"] == name]["dialogue"].values[0]
    )
)

fan_consensus_bb = {
    "Walter": "INTJ",
    "Jesse": "ESFP",
    "Skyler": "ISTJ",
    "Hank": "ESTP",
    "Walter Jr": "ISFP"
}
bb_dialogues["fan_consensus_mbti"] = bb_dialogues["character"].map(fan_consensus_bb)
bb_dialogues["show"] = "Breaking Bad"
print("Breaking Bad done!")

# ============ COMBINE & SAVE ============
final_df = pd.concat([
    character_dialogues[["character", "show", "predicted_mbti", "fan_consensus_mbti"]],
    bb_dialogues[["character", "show", "predicted_mbti", "fan_consensus_mbti"]]
], ignore_index=True)

final_df.to_csv("../data/final_results.csv", index=False)

print("\n✅ Final Results saved!")
print(final_df)
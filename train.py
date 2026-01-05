import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("Loading dataset...")

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only label + message
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Normalize labels
df["label"] = df["label"].str.lower()

print(f"Total messages: {len(df)}")
print(df["label"].value_counts())

# Train-test split (important for evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]  # VERY IMPORTANT for imbalance
)

# TF-IDF Vectorizer (better than CountVectorizer)
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

print("Training model...")
model.fit(X_train_vec, y_train)

# Evaluation
print("\nMODEL EVALUATION:")
y_pred = model.predict(X_test_vec)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("spam_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("\n✅ Model saved as spam_model.pkl")
print("🎯 Training complete!")

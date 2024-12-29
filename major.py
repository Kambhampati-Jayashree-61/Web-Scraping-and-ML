import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Simulation
data = {
    "Title": [f"Movie {i}" for i in range(1, 101)],
    "Description": [
        "An epic story of love and betrayal" if i % 3 == 0 else
        "A thrilling action-packed journey" if i % 3 == 1 else
        "A heartwarming tale of friendship"
        for i in range(1, 101)
    ],
    "Genre": [random.choice(["Action", "Drama", "Comedy"]) for _ in range(100)],
    "Release_Year": [random.randint(2000, 2023) for _ in range(100)],
    "Rating": [round(random.uniform(1.0, 5.0), 1) for _ in range(100)]
}

df = pd.DataFrame(data)

# 2. Data Preprocessing
label_encoder = LabelEncoder()
df["Genre_Encoded"] = label_encoder.fit_transform(df["Genre"])

# Features and Targets
X_genre = df[["Release_Year", "Rating"]]  # Features for classification
y_genre = df["Genre_Encoded"]             # Target for classification
X_rating = df[["Release_Year", "Genre_Encoded"]]  # Features for regression
y_rating = df["Rating"]                           # Target for regression

# Train-Test Split
X_genre_train, X_genre_test, y_genre_train, y_genre_test = train_test_split(X_genre, y_genre, test_size=0.2, random_state=42)
X_rating_train, X_rating_test, y_rating_train, y_rating_test = train_test_split(X_rating, y_rating, test_size=0.2, random_state=42)

# 3. Exploratory Data Analysis (EDA)
# Genre Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Genre", data=df)
plt.title("Genre Distribution")
plt.show()

# Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Rating"], kde=True, bins=10)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.show()

# 4. Genre Classification (Random Forest Classifier)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_genre_train, y_genre_train)
y_genre_pred = clf.predict(X_genre_test)

# Classification Evaluation
print("Genre Classification Report:")
print(classification_report(y_genre_test, y_genre_pred))
print("Classification Accuracy:", accuracy_score(y_genre_test, y_genre_pred))

# 5. Rating Prediction (Random Forest Regressor)
reg = RandomForestRegressor(random_state=42)
reg.fit(X_rating_train, y_rating_train)
y_rating_pred = reg.predict(X_rating_test)

# Regression Evaluation
mse = mean_squared_error(y_rating_test, y_rating_pred)
r2 = r2_score(y_rating_test, y_rating_pred)

print("\nRating Prediction Metrics:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 6. Visualization for Prediction Performance
# True vs Predicted Ratings
plt.figure(figsize=(8, 5))
plt.scatter(y_rating_test, y_rating_pred, alpha=0.7)
plt.title("True vs Predicted Ratings")
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.show()
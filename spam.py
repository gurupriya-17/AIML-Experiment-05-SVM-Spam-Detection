import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Dataset
data = pd.DataFrame({
    "text": [
        "Win a free lottery prize",
        "Congratulations you won money",
        "Meeting at 10 AM tomorrow",
        "Please submit the assignment",
        "You have won a free gift",
        "Project meeting is scheduled"
    ],
    "label": [1, 1, 0, 0, 1, 0]
})

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train SVM
model = SVC()
model.fit(X_train, y_train)

# Test new mail
mail = ["Congratulations! You won a free prize"]
mail_features = vectorizer.transform(mail)

prediction = model.predict(mail_features)

if prediction[0] == 1:
    print("Spam Mail")
else:
    print("Not Spam Mail")
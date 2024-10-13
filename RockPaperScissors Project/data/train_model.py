import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('hand_landmarks.csv')
X = data.iloc[:, 1:].values 
y = data.iloc[:, 0].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

with open('gesture_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'gesture_classifier.pkl'.")

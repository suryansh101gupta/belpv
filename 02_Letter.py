import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import shutil

import zipfile

with zipfile.ZipFile("letter+recognition.zip", 'r') as zip_ref:
    zip_ref.extractall()

df = pd.read_csv('letter-recognition.data')
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

X, y = df.drop('T', axis=1).values, df['T'].values

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

num_classes = 26

model = Sequential()
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

sample_idx = 8
sample_image = X_test[sample_idx].reshape(1, -1)
prediction = model.predict(sample_image)
predicted_letter = chr(65 + np.argmax(prediction)) # Convert index back to A-Z
actual_letter = chr(65 + y_test[sample_idx])

print(f"Predicted Letter: {predicted_letter}")
print(f"Actual Letter: {actual_letter}")
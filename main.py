import os
import pandas as pd
import numpy as np

# -----------------------------------------
# AUTO-DETECT THE DIRECTORY OF THIS SCRIPT
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to CSV file relative to this script
csv_path = os.path.join(BASE_DIR, "Churn_Modelling.csv")

print("Using CSV path:", csv_path)

# Load dataset
df = pd.read_csv(csv_path)

# -----------------------------------------
# Rest of your code
# -----------------------------------------

df.head()
df.shape
df.info()

df.drop(columns=['RowNumber','CustomerId','Surname'], inplace=True)

df = pd.get_dummies(df, columns=['Geography','Gender'], drop_first=True)

X = df.drop(columns=['Exited'])
y = df['Exited'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten


model = Sequential([
    Dense(11, activation='sigmoid', input_dim=X_train_trf.shape[1]),
    Dense(11, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_trf, y_train, batch_size=50, epochs=100, validation_split=0.2)

y_prob = model.predict(X_test_trf).ravel()
y_pred = (y_prob > 0.5).astype(int)

from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

**Credit Card Customer Churn Prediction (ANN)**

This project builds an Artificial Neural Network (ANN) to predict whether a bank customer will churn based on demographic and account-related features. The dataset is taken from the Churn Modelling dataset commonly used on Kaggle.

 **Project Overview**

Cleaned and prepared the dataset (removed ID columns, encoded categorical features, scaled numerical variables).

Built a 3-layer ANN using TensorFlow/Keras.

Trained the model for 100 epochs using the Adam optimizer and binary cross-entropy loss.

Evaluated the model on a test set and visualized accuracy and loss curves.

**Dataset**
File: Churn_Modelling.csv

Target: Exited (1 = churn, 0 = retained)

Dropped: RowNumber, CustomerId, Surname

Encoding: One-hot encoding on Geography and Gender

Scaling: StandardScaler applied to features

**Model Architecture**

Input layer: 11 neurons

Hidden layer 1: 11 neurons (sigmoid)

Hidden layer 2: 11 neurons (sigmoid)

Output layer: 1 neuron (sigmoid)

Loss: Binary Crossentropy

Optimizer: Adam

Epochs: 100

Batch size: 50

 **Model Performance**

Test Accuracy: Replace with your value → XX.XX%
(Computed using accuracy_score on test predictions.)

 **Visualizations Included**
 
Training vs Validation Accuracy

Training vs Validation Loss

These help understand model learning behaviour and signs of overfitting.

 **How to Run**

Install dependencies:

pip install numpy pandas scikit-learn tensorflow matplotlib


Place the dataset in:

Customer-Churn-Prediction/Churn_Modelling.csv


Run the Python script or notebook to train and evaluate the model.

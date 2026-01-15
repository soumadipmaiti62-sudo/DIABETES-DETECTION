import joblib
import numpy as np

# Load trained model
model = joblib.load("model/svm_model.pkl")

# Sample input data
input_data = np.array([[5, 116, 74, 0, 0, 25.6, 0.201, 30]])

# Make prediction
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Diabetic")
else:
    print("Not Diabetic")

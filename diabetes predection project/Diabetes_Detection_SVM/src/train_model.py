from sklearn.svm import SVC
import joblib
from src.data_preprocessing import preprocess_data
import os

X_train, X_test, y_train, y_test = preprocess_data()

model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/svm_model.pkl")

print("✅ Model trained and saved successfully")


import joblib
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_data

model = joblib.load("model/svm_model.pkl")
X_train, X_test, y_train, y_test = preprocess_data()

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

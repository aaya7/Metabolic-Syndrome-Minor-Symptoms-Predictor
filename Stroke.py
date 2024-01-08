import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stroke", page_icon="💉")

st.markdown("# Stroke Dataset 💉")
st.sidebar.header("Stroke")

st.image("be_fast_stroke.png", use_column_width=True, caption='You should know the signs of Strokes.')

st.write("""**👈Stroke is a serious medical emergency that can have severe consequences. By recognizing the signs and acting quickly, we can improve the chances of a successful recovery. Take preventive measures seriously and adopt a healthy lifestyle to reduce your risk of stroke. Together, we can spread awareness about stroke and make a difference in saving lives.**. 

Prevention is key. Leading a healthy lifestyle, including regular exercise, a balanced diet, and avoiding smoking and excessive alcohol, can significantly lower the risk of stroke. Managing chronic conditions, like high blood pressure and diabetes, is also essential. Stroke is a serious medical emergency that can have severe consequences. By recognizing the signs and acting quickly, we can improve the chances of a successful recovery. Take preventive measures seriously and adopt a healthy lifestyle to reduce your risk of stroke. Together, we can spread awareness about stroke and make a difference in saving lives.

Below show the Stroke Dataset that we are going to used.  
""")

stroke = pd.read_csv("dataset/stroke/healthcare-dataset-stroke-data.csv")
stroke

st.selectModel = st.selectbox("Select Model", options = ["Select Model Below: ", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbour"])

data_training = pd.read_csv("dataset/stroke/healthcare-dataset-stroke-data_training.csv")
data_training_input = data_training.drop(columns=["id","gender", "stroke", "work_type", "Residence_type", "smoking_status"], axis=1)
data_training_target = data_training['stroke']

data_testing = pd.read_csv("dataset/stroke/healthcare-dataset-stroke-data_testing.csv")
data_testing_input = data_testing.drop(columns=["id", "gender", "stroke", "work_type", "Residence_type", "smoking_status"], axis=1)
data_testing_target = data_testing['stroke']

if st.selectModel == "Naive Bayes":
	
	st.subheader("Naive Bayes.")

	# Create a Gaussian Naive Bayes classifier
	model = GaussianNB()

	# Train the model using the training data
	model.fit(data_training_input, data_training_target)
	y_pred = model.predict(data_testing_input)
	st.write("Successfully Trained the Model Using Naive Bayes")

	# Calculate the accuracy of the model's predictions
	accuracy = accuracy_score(data_testing_target, y_pred)
	st.write(f"Model Accuracy: {accuracy:.2f}")

elif st.selectModel == "Support Vector Machine":
    
	st.subheader("Support Vector Machine.")

	model = SVC(kernel='rbf')
	model.fit(data_training_input, data_training_target)
	y_pred = model.predict(data_testing_input)
	st.write("Successfully Trained the Model Using Support Vector Machine - RBF Kernel")
	
	accuracy = accuracy_score(data_testing_target, y_pred)
	st.write(f"Model Accuracy: {accuracy:.2f}")

elif st.selectModel == "K-Nearest Neighbour":

	st.subheader("K-Nearest Neighbour.")

	model = KNeighborsClassifier(n_neighbors=5)
	model.fit(data_training_input, data_training_target)
	y_pred = model.predict(data_testing_input)
	st.write("Successfully Trained the Model Using K-Nearest Neighbour - 10 Neighbors")

	accuracy = accuracy_score(data_testing_target, y_pred)
	st.write(f"Model Accuracy: {accuracy:.2f}")

import pandas as pd
import streamlit as st

# Load your trained machine learning model here (model should be defined and trained)
# model = ...

def predict_stroke(age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi):
    # Preprocess the input features to match the model's requirements
    input_data = pd.DataFrame({
	'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi]
    })
    
    # Use the trained model to predict diabetes
    prediction = model.predict(input_data)[0]
    return prediction

st.image("stroke_face.jpeg", use_column_width=True, caption='The example of face when strokes happened.')

def main():
    st.title("Stroke Prediction")
    st.write("Please provide your details to check if you probably have diabetes.")
    
    name = st.text_input("Name:")
    age = st.slider("Enter your Age:", min_value=10, max_value=100, value=20)
    hypertension = st.number_input("Hypertension status? 1 (Yes) | 0 (No):", min_value=0, max_value=1, step=1)
    heart_disease = st.number_input("Have a heart disease? 1 (Yes) | 0 (No):", min_value=0, max_value=1, step=1)
    ever_married = st.number_input("Do you ever married? 1 (Yes) | 0 (No):", min_value=0, max_value=1, step=1)
    avg_glucose_level = st.slider("Average of Glucose level (optional):", min_value=0.0, max_value=300.00, value=100.00)
    bmi = st.slider("BMI:", min_value=0.0, max_value=100.0, value=50.00)
    
    if st.button("Predict"):
        prediction = predict_stroke(age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi)
        
        if prediction == 1:
            st.write("Based on the input data, you probably do not have Stroke.")
        elif prediction == 0:
            st.write("Based on the input data, you probably have Stroke. I advice you to have a checkup as soon as possible. This also proved that you might have the minor symptoms of Metabolic Syndrome.")
        else:
            st.warning("Please enter the required details.")

st.image("strokes.jpg", use_column_width=True, caption='Strokes symptom for women and men.')

if __name__ == "__main__":
    main()

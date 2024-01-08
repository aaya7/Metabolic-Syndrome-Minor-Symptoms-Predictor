import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Obese", page_icon="🍔")

st.markdown("# Obese Dataset 🍔")
st.sidebar.header("Obese")

st.image("stomach.png", use_column_width=True)

st.write("""**👈Obesity is a chronic and complex medical condition characterized by an excessive accumulation of body fat that poses a significant health risk. It is a prevalent global health issue and is associated with numerous adverse effects on physical, emotional, and social well-being. The condition arises from a combination of genetic, environmental, and behavioral factors, with an imbalance between caloric intake and energy expenditure being a primary contributor. The World Health Organization (WHO) defines obesity based on a person's body mass index (BMI), which is calculated by dividing an individual's weight (in kilograms) by the square of their height (in meters). A BMI of 30 or above is considered obese, while a BMI between 25 and 29.9 indicates overweight**. 

Obesity can lead to a range of serious health complications, such as type 2 diabetes, heart disease, high blood pressure, stroke, certain cancers, joint problems, sleep apnea, and mental health issues. Additionally, obesity can have a profound impact on an individual's self-esteem and quality of life, often leading to social stigmatization and discrimination.

Addressing obesity requires a comprehensive approach, including lifestyle modifications, dietary changes, increased physical activity, and, in some cases, medical interventions. Preventive measures, such as promoting healthy eating habits, regular exercise, and awareness campaigns, are crucial in tackling the obesity epidemic and improving public health.

It is essential for individuals to recognize the potential risks of obesity and take proactive steps to maintain a healthy weight to reduce the associated health burdens and improve overall well-being.

Below show the Diabetes Dataset that we are going to used.  
""")

diabetes = pd.read_csv("dataset/obese/bodyfat.csv")
diabetes

selectModel = st.selectbox("Select Model", options = ["Select Model Below: ", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbour"])

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
train = pd.read_csv("dataset/obese/bodyfat_training.csv")
test = pd.read_csv("dataset/obese/bodyfat_testing.csv")

# Split the data into features (X) and target (y)
input_train = train.drop(columns=["Density", "Neck", "Chest", "Abdomen", "Knee", "Ankle", "Biceps","Forearm","Wrist","result"], axis=1)
input_test = test.drop(columns=["Density", "Neck", "Chest", "Abdomen", "Knee", "Ankle", "Biceps","Forearm","Wrist","result"], axis=1)
target_train = train["result"]
target_test = test["result"]

if selectModel == "Naive Bayes":
    
    st.subheader("Naive Bayes.")

    # Create and train the model
    model = GaussianNB()
    st.write("Training the Model...")
    model.fit(input_train, target_train)
    st.write("Successfully Trained the Model Using Naive Bayes")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
elif selectModel == "Support Vector Machine":
    
    st.subheader("Support Vector Machine.")
        
    model = SVC(kernel='linear')
    st.write("Training the Model...")
    model.fit(input_train, target_train)
    st.write("Successfully Trained the Model Using linear kernel")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
elif selectModel == "K-Nearest Neighbour":
    
    st.subheader("K-Nearest Neighbour.")
        
    model = KNeighborsClassifier(n_neighbors = 10)
    st.write("Training the Model...")
    model.fit(input_train, target_train)
    st.write("Successfully Trained the Model Using KNN")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
import pandas as pd
import streamlit as st

# Load your trained machine learning model here (model should be defined and trained)
# model = ...

def predict_obesity(BodyFat, Age, Weight, Height, Hip, Thigh):
    # Preprocess the input features to match the model's requirements
    input_data = pd.DataFrame({
        'BodyFat': [BodyFat],
	'Age': [Age],
        'Weight': [Weight],
        'Height': [Height],
        'Hip': [Hip],
        'Thigh': [Thigh]
    })
    
    # Use the trained model to predict diabetes
    prediction = model.predict(input_data)[0]
    return prediction

def main():
    st.title("Obesity Prediction")
    st.write("Please provide your details to check if you probably have obesity.")
    
    name = st.text_input("Name:")
    Age = st.number_input("Age:", min_value=0, max_value=150, step=1)
    BodyFat = st.number_input("Body Fat:", min_value=0.0, max_value=300.00, step=0.1)
    Weight = st.number_input("Weight:", min_value=0.0, max_value=300.00, step=0.1)
    Height = st.number_input("Height:", min_value=0.0, max_value=300.00, step=0.1)
    Hip = st.number_input("Hip:", min_value=0.0, max_value=300.00, step=0.1)
    Thigh = st.number_input("Thigh:", min_value=0.0, max_value=300.00, step=0.1)
    
    if st.button("Predict"):
    	prediction = predict_obesity(Age, BodyFat, Weight, Height, Hip, Thigh)
        
    	if prediction == 1:
       		st.write("Based on the input data, you probably have Obesity. I advice you to have a checkup as soon as possible. This also proved that you might have the minor symptoms of Metabolic Syndrome.")
    	elif prediction == 0:
        	st.write("Based on the input data, you probably do not have Obesity.")
    	else:
        	st.warning("Please enter the required details.")

if __name__ == "__main__":
    main()
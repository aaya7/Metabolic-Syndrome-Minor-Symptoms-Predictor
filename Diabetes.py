import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes", page_icon="🚨")

st.markdown("# Diabetes Dataset🚨")
st.sidebar.header("Diabetes")

st.image("Diabetes_share.jpg", use_column_width=True)

st.write("""**👈Diabetes is a chronic medical condition that affects how your body processes glucose, a type of sugar that serves as the primary source of energy for your cells. The hormone insulin plays a crucial role in regulating glucose levels in the bloodstream and helps facilitate the entry of glucose into the cells. The persistently high levels of glucose in the blood can lead to various health complications if left uncontrolled. These complications may affect the eyes (diabetic retinopathy), kidneys (diabetic nephropathy), nerves (diabetic neuropathy), and cardiovascular system (increased risk of heart disease and stroke). Proper management of diabetes through medications, insulin therapy, diet, exercise, and regular monitoring of blood sugar levels is essential to prevent or delay complications and maintain overall health and well-being. The symptoms of diabetes can include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision. If left unmanaged, diabetes can lead to serious complications, affecting various organs and systems in the body**. 

Long-term high blood sugar levels can damage blood vessels, nerves, and organs like the eyes, kidneys, heart, and feet. This can result in conditions such as diabetic retinopathy (eye damage), diabetic nephropathy (kidney damage), cardiovascular diseases, and diabetic neuropathy (nerve damage). To diagnose diabetes, doctors often perform blood tests to measure fasting blood sugar levels and hemoglobin A1c, which provides an average of blood sugar levels over the past few months. Early detection and proper management are essential to prevent complications and maintain a good quality of life for people with diabetes. Treatment plans may involve lifestyle changes, such as adopting a healthy diet and regular exercise, along with medications to control blood sugar levels. In recent years, advancements in diabetes management have been made, including the development of continuous glucose monitoring (CGM) systems, better insulin delivery methods, and improved diabetes education programs. However, diabetes remains a significant public health concern, and efforts to raise awareness, promote healthy lifestyles, and conduct ongoing research are essential in the global fight against this condition.
 
""")

st.write("Dataset description: The reason why the diabetes dataset was chosen is because this project needs this dataset to complete analysis regarding this syndrome. The data input for this dataset includes pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. The data target for this dataset outcome. Those are also the attributes of this dataset.")

st.write("Below show the Diabetes Dataset that we are going to used.")

diabetes = pd.read_csv("dataset/diabetes/diabetes.csv")
diabetes

st.selectModel = st.selectbox("Select Model", options = ["Select Model Below: ", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbour"])

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
train = pd.read_csv("dataset/diabetes/diabetes_training.csv")
test = pd.read_csv("dataset/diabetes/diabetes_testing.csv")

# Split the data into features (X) and target (y)
input_train = train.drop(columns="Outcome", axis=1)
input_test = test.drop(columns="Outcome", axis=1)
target_train = train["Outcome"]
target_test = test["Outcome"]

if st.selectModel == "Naive Bayes":
    
    st.subheader("Naive Bayes.")

    # Create and train the model
    model = GaussianNB()
    st.write("Training the Model...")
    model.fit(input_train, target_train)
    st.write("Successfully Trained the Model Using Naive Bayes")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
elif st.selectModel == "Support Vector Machine":
    
    st.subheader("Support Vector Machine.")
        
    model = SVC(kernel='linear')
    st.write("Training the Model...")
    model.fit(input_train, target_train)
    st.write("Successfully Trained the Model Using SVM - Linear Kernel")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
elif st.selectModel == "K-Nearest Neighbour":
    
    st.subheader("K-Nearest Neighbour.")
        
    model = KNeighborsClassifier(n_neighbors = 10)
    st.write("Training the Model...")
    model.fit(input_train, target_train)
    st.write("Successfully Trained the Model Using KNN - 10 Neighbors")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
import pandas as pd
import streamlit as st

# Load your trained machine learning model here (model should be defined and trained)
# model = ...

def predict_diabetes(age, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf):
    # Preprocess the input features to match the model's requirements
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })
    
    # Use the trained model to predict diabetes
    prediction = model.predict(input_data)[0]
    return prediction

st.image("diabetes.jpg", use_column_width=True)

def main():
    st.title("Diabetes Prediction")
    st.write("Please provide your details to check if you probably have diabetes.")
    
    name = st.text_input("Name:")
    age = st.number_input("Age:", min_value=0, max_value=150, step=1)
    pregnancies = st.number_input("Pregnancy number:", min_value=0, max_value=300, step=1)
    glucose = st.number_input("Glucose Level:", min_value=0, max_value=400, step=1)
    blood_pressure = st.number_input("Blood Pressure Level:", min_value=0, max_value=300, step=1)
    skin_thickness = st.number_input("Skin Thickness (optional):", min_value=0, max_value=400, step=1)
    insulin = st.number_input("Insulin (optional):", min_value=0, max_value=300, step=1)
    bmi = st.number_input("BMI:", min_value=0.0, max_value=100.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function (optional):", min_value=0.0, max_value=100.0, step=0.1)
    
    if st.button("Predict"):
        prediction = predict_diabetes(age, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf)
        
        if prediction == 1:
            st.write("Based on the input data, you probably have diabetes. This also proved that you might have the minor symptoms of Metabolic Syndrome.")
        elif prediction == 0:
            st.write("Based on the input data, you probably do not have diabetes.")
        else:
            st.warning("Please enter the required details.")

if __name__ == "__main__":
    main()



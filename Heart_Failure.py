import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Failure", page_icon="🫀")

st.markdown("# Heart Failure Dataset🫀")
st.sidebar.header("Heart Failure")

st.image("Heartfailure.jpg", use_column_width=True)

st.write("""**👈Heart failure is a serious medical condition where the heart is unable to pump blood effectively, leading to insufficient oxygen and nutrient delivery to the body's tissues. It often develops gradually, and common symptoms include shortness of breath, fatigue, swelling in the legs, and persistent coughing. Heart failure can result from various factors, such as coronary artery disease, high blood pressure, or damage to the heart muscles. It affects people of all ages but is more prevalent in older adults. Treatment involves lifestyle changes, medications, and, in some cases, surgery or medical devices to improve heart function and manage symptoms. Early detection and management are crucial to enhance quality of life and reduce complications.**. 

The causes of heart failure can be diverse and may include conditions that damage or strain the heart, such as coronary artery disease, high blood pressure, heart attacks, or heart valve problems. Other factors like diabetes, obesity, and certain medications can also contribute to its development. Prevention involves adopting a heart-healthy lifestyle, including regular exercise, a balanced diet low in salt and saturated fats, managing conditions like diabetes and high blood pressure, avoiding smoking and excessive alcohol consumption, and seeking early medical attention for any heart-related symptoms. Early detection and proactive management of risk factors can significantly reduce the risk of heart failure and promote overall cardiovascular health.

Below show the Heart Failure Dataset that we are going to used.  
""")

heartF = pd.read_csv("dataset/heartFailure/heartfailure.csv")
heartF

selectModel = st.selectbox("Select Model", options = ["Select Model Below: ", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbour"])

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
train = pd.read_csv("dataset/heartFailure/heartfailure_train.csv")
test = pd.read_csv("dataset/heartFailure/heartfailure_test.csv")

# Split the data into features (X) and target (y)
input_train = train.drop(columns=["Sex","ChestPainType","RestingECG","HeartDisease"], axis=1)
input_test = test.drop(columns=["Sex","ChestPainType","RestingECG","HeartDisease"], axis=1)
target_train = train["HeartDisease"]
target_test = test["HeartDisease"]

if selectModel == "Naive Bayes":
    
    st.subheader("Naive Bayes.")

    # Create and train the model
    model = GaussianNB()
    st.write("Training the Model...")
    model.fit(input_train,target_train)
    st.write("Successfully Trained the Model Using Naive Bayes")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
elif selectModel == "Support Vector Machine":
    
    st.subheader("Support Vector Machine.")
        
    model = SVC(kernel='linear')
    st.write("Training the Model...")
    model.fit(input_train,target_train)
    st.write("Successfully Trained the Model Using SVM")

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

def predict_heartfailure(Age,RestingBP,MaxHR,ExerciseA0gi0a,Cholesterol):
    # Preprocess the input features to match the model's requirements
    input_data = pd.DataFrame({
        'Age': [Age],
        'RestingBP': [RestingBP],
        'MaxHR': [MaxHR],
	'ExerciseA0gi0a': [ExerciseA0gi0a],
        'Cholesterol': [Cholesterol]
  
       
    })
    
    # Use the trained model to predict diabetes
    prediction = model.predict(input_test)[0]
    return prediction

st.image("symptom-heart-faliure.jpg", use_column_width=True)

def main():
    st.title("Heart failure Prediction")
    st.write("Please provide your details to check if you probably have diabetes.")
    
    name = st.text_input("Name:")
    Age = st.number_input("Age:", min_value=0, max_value=150, step=1)
    RestingBP = st.number_input("RestingBP:", min_value=0, max_value=200, step=1)
    MaxHR = st.number_input("MaxHR:", min_value=0, max_value=400, step=1)
    Cholesterol = st.number_input("Cholesterol:", min_value=0, max_value=300, step=1)
    ExerciseA0gi0a = st.number_input("Exercise Yes(1) No(0)):", min_value=0, max_value=1, step=1)


    
    if st.button("Predict"):
        prediction = predict_heartfailure(Age,RestingBP,MaxHR,ExerciseA0gi0a,Cholesterol)
  
        if prediction == 1:
            st.write("Based on the input data, you probably have Heart Failure. This also proved that you might have the minor symptoms of Metabolic Syndrome.")
        elif prediction == 0:
            st.write("Based on the input data, you probably do not have Heart Failure.")
        else:
            st.warning("Please enter the required details.")

if __name__ == "__main__":
    main()
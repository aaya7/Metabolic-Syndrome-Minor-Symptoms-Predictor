import streamlit as st
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Attack", page_icon="🫀")

st.markdown("# Heart Attack Dataset🫀")
st.sidebar.header("Heart Attack")

st.image("HeartAttack.png", use_column_width=True)

st.write("""**👈A heart attack, also known as a myocardial infarction (MI), occurs when there is a sudden blockage of blood flow to a part of the heart muscle, leading to damage or death of heart cells due to a lack of oxygen and nutrients. It is a medical emergency that requires immediate attention and treatment. Heart attacks can vary in severity, from mild to severe, and can be life-threatening.**. 

Causes:
Heart attacks are usually caused by the buildup of fatty deposits, cholesterol, and other substances (collectively known as plaque) on the walls of the coronary arteries, which supply blood to the heart muscle. When a plaque ruptures or a blood clot forms, it can block the artery and reduce or stop blood flow to the heart.

Symptoms of a heart attack typically include chest pain or discomfort, characterized by a sensation of pressure, tightness, heaviness, or squeezing in the chest. The pain may radiate to the arms, back, neck, jaw, or stomach. Shortness of breath, cold sweat, and feelings of nausea or vomiting may also be present. Some individuals may experience atypical symptoms, particularly women, older adults, and those with diabetes. It is crucial to seek immediate medical attention if any of these symptoms occur, as a heart attack is a medical emergency that requires prompt treatment to restore blood flow to the heart and prevent further damage or potential life-threatening complications.

Below show the Heart Attack Dataset that we are going to used.  
""")

heartF = pd.read_csv("dataset/heartAttack/heart.csv")
heartF

selectModel = st.selectbox("Select Model", options = ["Select Model Below: ", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbour"])

# Load the dataset (replace 'your_dataset.csv' with your actual dataset file)
train = pd.read_csv("dataset/heartAttack/heart_training.csv")
test = pd.read_csv("dataset/heartAttack/heart_testing.csv")

# Split the data into features (X) and target (y)
input_train = train.drop(columns=["sex", "fbs", "restecg", "exng", "slp", "caa", "thall", "output"], axis=1)
input_test = test.drop(columns=["sex", "fbs", "restecg", "exng", "slp", "caa", "thall", "output"], axis=1)
target_train = train["output"]
target_test = test["output"]  

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
    st.write("Successfully Trained the Model Using SVM which is Linear Kernel")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
elif selectModel == "K-Nearest Neighbour":
    
    st.subheader("K-Nearest Neighbour.")
        
    model = KNeighborsClassifier(n_neighbors = 1)
    st.write("Training the Model...")
    model.fit(input_train, target_train)
    st.write("Successfully Trained the Model Using KNN-Neighbors 1")

    # Calculate the accuracy on the test set
    accuracy = accuracy_score(target_test, model.predict(input_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
import pandas as pd
import streamlit as st

# Load your trained machine learning model here (model should be defined and trained)
# model = ...

def predict_heartAttack(age, cp, trtbps, chol, thalachh, oldpeak):
    # Preprocess the input features to match the model's requirements
    input_data = pd.DataFrame({
        'age': [age],
        'cp': [cp],
        'trtbps': [trtbps],
        'chol': [chol],
        'thalachh': [thalachh],
        'oldpeak': [oldpeak]
    })
    
    # Use the trained model to predict diabetes
    prediction = model.predict(input_data)[0]
    return prediction

def main():
    st.title("Heart Attack Prediction")
    st.write("Please provide your details to check if you probably have Heart Attack.")
    
    name = st.text_input("Name:")
    age = st.number_input("Age:", min_value=0, max_value=150, step=1)
    cp = st.number_input("Chest Pain:", min_value=0, max_value=300, step=1)
    trtbps = st.number_input("Resting Blood Pressure:", min_value=0, max_value=400, step=1)
    chol = st.slider("Cholestoral:", min_value=0, max_value=500, value=150)
    thalachh = st.number_input("Maximum Heart Rate Achieved:", min_value=0, max_value=400, step=1)
    oldpeak = st.number_input("ST Depression Induced:", min_value=0.0, max_value=100.0, step=0.1)
    
    if st.button("Predict"):
        prediction = predict_heartAttack(age, cp, trtbps, chol, thalachh, oldpeak)
        
        if prediction == 1:
            st.write("Based on the input data, you probably have Heart Attack. This also proved that you might have the minor symptoms of Metabolic Syndrome.")
        elif prediction == 0:
            st.write("Based on the input data, you probably do not have Heart Attack.")
        else:
            st.warning("Please enter the required details.")

if __name__ == "__main__":
    main()
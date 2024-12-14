import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model components
@st.cache_resource
def load_model_components():
    model = joblib.load("./models/random_forest_model.joblib")
    mlb = joblib.load("./models/multilabel_binarizer.joblib")
    svd = joblib.load("./models/svd_transformer.joblib")
    skill_names = pd.read_csv("./models/skill_names.csv")["0"].tolist()
    return model, mlb, svd, skill_names

def predict_cs_role(skills, education_level, model, mlb, svd):
    # Convert skills to binary format
    skills_binary = mlb.transform([skills])
    
    # Apply SVD transformation
    skills_reduced = svd.transform(skills_binary)
    
    # Combine with education level
    X = np.hstack([skills_reduced, [[education_level]]])
    
    # Make prediction
    prediction = model.predict(X)
    probability = model.predict_proba(X)[0]
    
    return prediction[0], probability

# Streamlit UI
st.title("CS Role Predictor")

# Load model components
model, mlb, svd, available_skills = load_model_components()

# Create the form
with st.form("prediction_form"):
    # Multi-select for skills
    selected_skills = st.multiselect(
        "Select your skills",
        options=available_skills,
        help="Choose all skills that apply"
    )
    
    # Education level selector
    education_level = st.selectbox(
        "Select your highest CS education level",
        options=[
            ("None", 0),
            ("Associate's Degree", 1),
            ("Bachelor's Degree", 2),
            ("Master's Degree", 3),
            ("PhD", 4)
        ],
        format_func=lambda x: x[0]
    )[1]  # Get the numeric value
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# Make prediction when form is submitted
if submitted:
    if not selected_skills:
        st.warning("Please select at least one skill")
    else:
        # Make prediction
        prediction, probability = predict_cs_role(
            selected_skills,
            education_level,
            model,
            mlb,
            svd
        )
        
        # Display results
        st.header("Prediction Results")
        
        if prediction == 1:
            confidence = probability[1] * 100
            st.success(f"You are likely to be in a CS role! (Confidence: {confidence:.1f}%)")
        else:
            confidence = probability[0] * 100
            st.info(f"You are likely to be in a non-CS role. (Confidence: {confidence:.1f}%)")
        
        # Display probability distribution
        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Role Type': ['Non-CS Role', 'CS Role'],
            'Probability': probability
        })
        st.bar_chart(prob_df.set_index('Role Type'))
        
        # Display selected skills
        st.subheader("Your Selected Skills")
        st.write(", ".join(selected_skills))
        
        # Display education level
        st.subheader("Your Education Level")
        education_mapping = {0: "None", 1: "Associate's", 2: "Bachelor's", 3: "Master's", 4: "PhD"}
        st.write(education_mapping[education_level])
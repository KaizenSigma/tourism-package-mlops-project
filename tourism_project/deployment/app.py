import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "tourism_project/deployment/best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Tourism Package Prediction", layout="centered")
st.title("Tourism Package Purchase Prediction")
st.write("Enter customer details below to predict whether the package is likely to be purchased.")

model = load_model()

age = st.number_input("Age", min_value=18, max_value=90, value=35)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch", min_value=1, max_value=180, value=15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=3)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.number_input("Number Of Trips", min_value=0, max_value=20, value=2)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own Car", [0, 1])
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Manager", "Senior Manager", "AVP", "VP", "Executive"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=300000, value=30000)

if st.button("Predict"):
    input_df = pd.DataFrame(
        [{
            "Age": age,
            "TypeofContact": type_of_contact,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": number_of_person_visiting,
            "NumberOfFollowups": number_of_followups,
            "ProductPitched": product_pitched,
            "PreferredPropertyStar": preferred_property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": number_of_children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
        }]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Prediction: Customer is likely to purchase the package (Probability: {probability:.2%})")
    else:
        st.warning(f"Prediction: Customer is unlikely to purchase the package (Probability: {probability:.2%})")

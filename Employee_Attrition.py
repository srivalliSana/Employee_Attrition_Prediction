import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------------
# Load data and model
# ------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition (1).csv")
model = joblib.load("Employee_Attrition_model.pkl")

# ------------------------------
# Setup
# ------------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("         💼 Employee Attrition Prediction System         ")

# ------------------------------
# Sidebar Navigation
# ------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "EDA - Correlation", "EDA - Distributions", "EDA - Trends", "Predict Attrition"]
)

# ------------------------------
# Home Page
# ------------------------------
if menu == "🏠 Home":
    st.header("Welcome to the Employee Attrition Prediction System")

    st.markdown("""
    This project aims to predict the risk of employee attrition (employees leaving the company) 
    using machine learning techniques. By analyzing HR data, we can identify patterns and 
    proactively work to improve employee retention.

    **About the Dataset:**
    - IBM HR Analytics Employee Attrition Dataset
    - ~35+ features: employee demographics, satisfaction, environment, etc.
    - Target variable: **Attrition (Yes/No)**
    - Available on Kaggle
    """)

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🔎 Dataset Statistics")
    st.dataframe(df.describe())

    st.subheader("📝 Dataset Info")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(info_df)


# ------------------------------
# EDA - Correlation Page
# ------------------------------
elif menu == "EDA - Correlation":
    st.header("🔎 Correlation Analysis")
    st.write("Correlation heatmap between numerical HR features.")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    st.write("**Top correlated pairs:**")
    corr_matrix = df.select_dtypes(include=np.number).corr()
    high_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
    high_corr = high_corr[(high_corr < 1.0) & (high_corr > 0.5)].drop_duplicates()
    st.dataframe(high_corr)

# ------------------------------
# EDA - Distributions
# ------------------------------
elif menu == "EDA - Distributions":
    st.header("📊 Attrition and Feature Distributions")

    st.subheader("Attrition Breakdown")
    attr_counts = df["Attrition"].value_counts()
    fig, ax = plt.subplots()
    attr_counts.plot(kind="bar", color=["green","red"], ax=ax)
    ax.set_ylabel("Number of Employees")
    ax.set_title("Attrition Distribution")
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Monthly Income Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df["MonthlyIncome"], bins=30, kde=True, ax=ax3)
    st.pyplot(fig3)

    st.subheader("Years at Company")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=df, x="YearsAtCompany", ax=ax4)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)
    st.pyplot(fig4)

# ------------------------------
# EDA - Trends
# ------------------------------
elif menu == "EDA - Trends":
    st.header("📈 Trends and Relationships")

    st.subheader("Attrition by Department")
    fig5, ax5 = plt.subplots()
    sns.countplot(data=df, x="Department", hue="Attrition", ax=ax5)
    st.pyplot(fig5)

    st.subheader("Attrition by Business Travel")
    fig6, ax6 = plt.subplots()
    sns.countplot(data=df, x="BusinessTravel", hue="Attrition", ax=ax6)
    st.pyplot(fig6)

    st.subheader("Attrition by Job Role")
    fig7, ax7 = plt.subplots(figsize=(10,5))
    sns.countplot(data=df, x="JobRole", hue="Attrition", ax=ax7)
    ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45)
    st.pyplot(fig7)

    st.subheader("Job Satisfaction vs Attrition")
    fig8, ax8 = plt.subplots()
    sns.boxplot(data=df, x="Attrition", y="JobSatisfaction", ax=ax8)
    st.pyplot(fig8)

# ------------------------------
# Prediction Page
# ------------------------------
elif menu == "Predict Attrition":
    st.header("🎯 Employee Attrition Prediction")

    st.markdown("Provide employee details below to predict their attrition risk.")

    # numeric inputs
    age = st.slider("Age", 18, 60, 30)
    daily_rate = st.number_input("Daily Rate", value=800)
    distance_from_home = st.slider("Distance From Home (km)", 1, 50, 5)
    education = st.selectbox("Education Level", [1,2,3,4,5], index=2)
    environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1,4,3)
    hourly_rate = st.number_input("Hourly Rate", value=80)
    job_involvement = st.slider("Job Involvement (1-4)", 1,4,3)
    job_level = st.selectbox("Job Level", [1,2,3,4,5], index=1)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1,4,3)
    monthly_income = st.number_input("Monthly Income", value=5000)
    monthly_rate = st.number_input("Monthly Rate", value=15000)
    num_companies_worked = st.number_input("Num Companies Worked", value=1)
    percent_salary_hike = st.slider("Percent Salary Hike", 10,50,15)
    performance_rating = st.selectbox("Performance Rating", [1,2,3,4], index=2)
    relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1,4,3)
    stock_option_level = st.selectbox("Stock Option Level", [0,1,2,3], index=0)
    total_working_years = st.number_input("Total Working Years", value=5)
    training_times_last_year = st.slider("Training Times Last Year", 0,10,2)
    work_life_balance = st.selectbox("Work Life Balance (1-4)", [1,2,3,4], index=2)
    years_at_company = st.number_input("Years At Company", value=3)
    years_in_current_role = st.number_input("Years In Current Role", value=2)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", value=1)
    years_with_curr_manager = st.number_input("Years With Current Manager", value=2)

    # categorical inputs with basic mapping
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely","Travel_Frequently","Non-Travel"])
    department = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])
    education_field = st.selectbox("Education Field", ['Life Sciences','Medical','Marketing','Technical Degree','Other'])
    gender = st.selectbox("Gender", ["Male","Female"])
    marital_status = st.selectbox("Marital Status", ["Single","Married","Divorced"])
    over_time = st.selectbox("OverTime", ["Yes","No"])

    # mappings
    business_travel_map = {"Travel_Rarely":0, "Travel_Frequently":1, "Non-Travel":2}
    department_map = {"Sales":0, "Research & Development":1, "Human Resources":2}
    education_field_map = {"Life Sciences":0, "Medical":1, "Marketing":2, "Technical Degree":3,"Other":4}
    gender_map = {"Male":0,"Female":1}
    marital_map = {"Single":0,"Married":1,"Divorced":2}
    overtime_flag = 1 if over_time=="Yes" else 0

    # static fields (from dataset)
    employee_count = 1
    employee_number = 1
    over18 = 1
    standard_hours = 80

    # job role one-hot (9 roles)
    job_roles = ['Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager',
                 'Manufacturing Director', 'Research Director', 'Research Scientist',
                 'Sales Executive', 'Sales Representative']
    job_role = st.selectbox("Job Role", job_roles)
    job_role_onehot = [1 if job_role==jr else 0 for jr in job_roles]

    # build the final feature array
    features = np.array([
        age,
        business_travel_map[business_travel],
        daily_rate,
        department_map[department],
        distance_from_home,
        education,
        education_field_map[education_field],
        employee_count,
        employee_number,
        environment_satisfaction,
        gender_map[gender],
        hourly_rate,
        job_involvement,
        job_level,
        job_satisfaction,
        marital_map[marital_status],
        monthly_income,
        monthly_rate,
        num_companies_worked,
        over18,
        overtime_flag,
        percent_salary_hike,
        performance_rating,
        relationship_satisfaction,
        standard_hours,
        stock_option_level,
        total_working_years,
        training_times_last_year,
        work_life_balance,
        years_at_company,
        years_in_current_role,
        years_since_last_promotion,
        years_with_curr_manager,
        *job_role_onehot  # unpack 9 job role flags
    ]).reshape(1,-1)

    if st.button("🔍 Predict"):
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        if prediction == 1 or prediction=="Yes":
            st.error(f"⚠️ High Attrition Risk Detected! (Probability: {prob:.2%})")
        else:
            st.success(f"✅ Low Attrition Risk Detected. (Probability: {prob:.2%})")

        st.markdown(f"**Model Confidence: {prob:.2%} that the employee will leave.**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")


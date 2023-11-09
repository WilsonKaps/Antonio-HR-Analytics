import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import json
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go

def load_lottiefile(filepath: str):
    with open(filepath) as f:
        data = json.load(f)
        return data
    

place = "C:/Users/wilso/OneDrive/Desktop/Work/testing/code.json"
lottie_coding = load_lottiefile(place)
st_lottie(place, key="hello")    

st.title("Antonio Consultancy Firm")

df = pd.DataFrame()

page = st.sidebar.radio("Navigation", ("Descriptive Analysis", "HR Insights", "Predictions"))


uploaded_file = st.sidebar.file_uploader('Upload the file', type=["csv", "xlsx"])
if uploaded_file is not None:
   
    df = pd.read_csv(uploaded_file) 
    title_text = st.empty()
    for _ in range(4):
        title_text.text("Antonio HR Analytics")
        time.sleep(0.1)
        title_text.text("Antonio HR Analytics.")
        time.sleep(0.1)
        title_text.text("Antonio HR Analytics..")
        time.sleep(0.1)

    title_text.text("Antonio HR Analytics")

    department_mapping = {
        'Sales': 0,
        'Research & Development': 1,
        'Human Resources': 2
    }

    df['DepartmentEncoded'] = df['Department'].map(department_mapping)

    if page == "Descriptive Analysis":
        st.header("Descriptive Analysis")

        fig_department = px.histogram(df, x='DepartmentEncoded', nbins=3, labels={'DepartmentEncoded': 'Department'}, title='Number of People in Each Department')
        st.plotly_chart(fig_department)
        st.subheader("Number of People in Each Department")
        

        st.subheader("Employee Age Distribution")
        age_slider = st.slider("Select Age Range", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(int(df['Age'].min()), int(df['Age'].max())))
        fig_age = px.histogram(df[(df['Age'] >= age_slider[0]) & (df['Age'] <= age_slider[1])], x='Age', nbins=20, labels={'Age': 'Employee Age'}, title='Employee Age Distribution')
        st.plotly_chart(fig_age)

        gender_counts = df['Gender'].value_counts()
        st.subheader("Gender Distribution")
        fig_gender = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values)])
        st.plotly_chart(fig_gender)

        colors = ['#66b3ff', '#99ff99']
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))

      
    elif page == "HR Insights":
        st.header("HR Insights")

        if 'Attrition' in df.columns:
            st.subheader("Job Role Distribution")
            fig = px.bar(df, x=df['JobRole'].value_counts().index, y=df['JobRole'].value_counts().values, labels={'x':'Job Role', 'y':'Count'}, title='Job Role Distribution')
            st.plotly_chart(fig)

            st.subheader("Education Field Distribution")
            fig = px.bar(df, x=df['EducationField'].value_counts().index, y=df['EducationField'].value_counts().values, labels={'x':'Education Field', 'y':'Count'}, title='Education Field Distribution')
            st.plotly_chart(fig)

            st.subheader("Age Distribution by Attrition")
            fig = px.histogram(df, x='Age', color='Attrition', nbins=20, labels={'x':'Age', 'y':'Count'}, title='Age Distribution by Attrition')
            st.plotly_chart(fig)

        else:
            st.warning("Attrition column not found in the dataset. Update the data source as needed.")

    elif page == "Predictions":
        st.header("Predictions")
        df['BusinessTravel'] = df['BusinessTravel'].apply(lambda x: 0 if x == 'Travel_Rarely' else (1 if x == 'Non_Travel' else 2))
        df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x in ['Yes', 'Y'] else 0)

        education_field_mapping = {
            'Life Sciences': 0,
            'Medical': 1,
            'Marketing': 2,
            'Technical Degree': 3,
            'Human Resources': 4,
            'Other': 5
        }

        df['EducationFieldEncoded'] = df['EducationField'].map(education_field_mapping)
        gender_mapping = {
            'Male': 0,
            'Female': 1
        }

        df['GenderEncoded'] = df['Gender'].map(gender_mapping)

        job_role_mapping = {
            'Sales Executive': 0,
            'Research Scientist': 1,
            'Laboratory Technician': 2,
            'Manufacturing Director': 3,
            'Healthcare Representative': 4,
            'Manager': 5,
            'Sales Representative': 6,
            'Research Director': 7,
            'Human Resources': 8
        }

        df['JobRoleEncoded'] = df['JobRole'].map(job_role_mapping)

        marital_status_mapping = {
            'Single': 0,
            'Married': 1,
            'Divorced': 2
        }

        df['MaritalStatusEncoded'] = df['MaritalStatus'].map(marital_status_mapping)

        over_time_mapping = {
            'Yes': 0,
            'No': 1
        }

        df['OverTimeEncoded'] = df['OverTime'].map(over_time_mapping)
        X = df[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'DepartmentEncoded', 'EducationFieldEncoded', 'GenderEncoded', 'JobRoleEncoded', 'MaritalStatusEncoded', 'OverTimeEncoded']]
        y = df['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")

        st.subheader("Confusion Matrix")
        confusion = confusion_matrix(y_test, y_pred)
        fig = px.imshow(confusion, labels=dict(x="Predicted", y="Actual", color="Count"), x=['No', 'Yes'], y=['No', 'Yes'], title='Confusion Matrix', color_continuous_scale="Blues")
        st.plotly_chart(fig)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        
        st.subheader("ROC Curve")
        fig = px.line(x=fpr, y=tpr, labels={'x':'False Positive Rate', 'y':'True Positive Rate'}, title='Receiver Operating Characteristic (ROC)', line_shape='linear')
        fig.update_traces(line=dict(color='blue', width=2))  
        fig.add_shape(type='line', line=dict(dash='dash', color='white'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig)

if uploaded_file is None:

    
    title_text = st.empty()
    for _ in range(4):
        title_text.text("Antonio HR Analytics")
        time.sleep(0.1)
        title_text.text("Antonio HR Analytics.")
        time.sleep(0.1)
        title_text.text("Antonio HR Analytics..")
        time.sleep(0.1)

    st.info("Please upload your file", icon="🤖")
    st_lottie(lottie_coding)

    
    title_text.text("Antonio HR Analytics")

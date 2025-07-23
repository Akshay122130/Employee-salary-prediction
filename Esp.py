import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Page configuration
st.set_page_config(page_title="Salary Prediction", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:",
                           ["Home", "Data Exploration", "Model Training", "Prediction"])


# Load the dataset
@st.cache_data
def load_data():
    """Load and return the dataset"""
    try:
        data = pd.read_csv(r"C:\Users\madet\Downloads\adult\adult.csv")
        data.replace('?', np.nan, inplace=True)
        data.fillna(data.mode().iloc[0], inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Data Preprocessing
def preprocess_data(data):
    """Clean and preprocess the data"""
    # Feature engineering
    data['capital-total'] = data['capital-gain'] - data['capital-loss']
    data['hours-per-week-cat'] = pd.cut(data['hours-per-week'],
                                        bins=[0, 30, 40, 50, 100],
                                        labels=['part-time', 'full-time', 'overtime', 'workaholic'])

    # Encode categorical variables
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country', 'hours-per-week-cat']

    le = LabelEncoder()
    for col in categorical_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])

    # Drop unnecessary columns
    cols_to_drop = ['fnlwgt', 'education', 'capital-gain', 'capital-loss']
    data.drop([col for col in cols_to_drop if col in data.columns], axis=1, inplace=True)

    # Handle outliers
    numeric_cols = ['age', 'educational-num', 'hours-per-week', 'capital-total']
    for col in numeric_cols:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = np.where(data[col] > upper_bound, upper_bound,
                                 np.where(data[col] < lower_bound, lower_bound, data[col]))

    return data


# Model Training
def train_model(data):
    """Train and evaluate the model"""
    X = data.drop('income', axis=1)
    y = data['income']

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Train model with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_res)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    return best_model, scaler, le, X_test, y_test, y_pred


def predict_salary(input_data, model, scaler, le):
    """Make a prediction with consistent features"""
    # Create DataFrame with all expected columns
    expected_features = [
        'age', 'workclass', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender',
        'hours-per-week', 'native-country', 'capital-total',
        'hours-per-week-cat'
    ]

    # Create empty DataFrame with expected columns
    input_df = pd.DataFrame(columns=expected_features)

    # Fill in available features
    for col in input_data:
        if col in expected_features:
            input_df[col] = [input_data[col]]

    # Calculate derived features
    if 'capital-gain' in input_data and 'capital-loss' in input_data:
        input_df['capital-total'] = input_data['capital-gain'] - input_data['capital-loss']

    if 'hours-per-week' in input_data:
        # Create hours-per-week-cat (same way as during training)
        hours = input_data['hours-per-week']
        if hours <= 30:
            input_df['hours-per-week-cat'] = 'part-time'
        elif hours <= 40:
            input_df['hours-per-week-cat'] = 'full-time'
        elif hours <= 50:
            input_df['hours-per-week-cat'] = 'overtime'
        else:
            input_df['hours-per-week-cat'] = 'workaholic'

    # Fill missing values with defaults (same as training)
    input_df.fillna({
        'native-country': 'United-States',  # Most common value
        'capital-total': 0,
        'hours-per-week-cat': 'full-time'
    }, inplace=True)

    # Encode categorical variables (same as during training)
    categorical_cols = ['workclass', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country',
                        'hours-per-week-cat']

    for col in categorical_cols:
        if col in input_df.columns:
            # Use the same LabelEncoder that was fit during training
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            input_df[col] = input_df[col].map(le_dict).fillna(0).astype(int)

    # Ensure columns are in correct order
    input_df = input_df[expected_features]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    return prediction, prediction_proba


# Home Page
if options == "Home":
    st.title("Employee Salary Prediction")
    st.image("https://images.unsplash.com/photo-1579621970563-ebec7560ff3e", width=700)
    st.markdown("""
    ## Welcome to the Salary Prediction App

    This application predicts whether an employee's income exceeds $50K/year based on census data.

    **Navigate using the sidebar** to:
    - Explore the dataset
    - View model training results
    - Make predictions with our trained model
    """)

# Data Exploration Page
elif options == "Data Exploration":
    st.title("Data Exploration")
    data = load_data()

    if data is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(data.head())

        st.subheader("Dataset Summary")
        st.write(data.describe())

        st.subheader("Visualizations")

        # Income Distribution
        st.write("### Income Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='income', data=data, ax=ax)
        st.pyplot(fig)

        # Age Distribution by Income
        st.write("### Age Distribution by Income")
        fig, ax = plt.subplots()
        sns.boxplot(x='income', y='age', data=data, ax=ax)
        st.pyplot(fig)

        # Correlation Matrix
        st.write("### Correlation Matrix")
        numeric_data = data.select_dtypes(include=['int64', 'float64'])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Model Training Page
elif options == "Model Training":
    st.title("Model Training")
    data = load_data()

    if data is not None:
        with st.spinner("Preprocessing data..."):
            processed_data = preprocess_data(data)

        st.subheader("Processed Data Preview")
        st.dataframe(processed_data.head())

        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes"):
                model, scaler, le, X_test, y_test, y_pred = train_model(processed_data)

            st.success("Model trained successfully!")

            # Save model artifacts
            joblib.dump(model, 'salary_predictor_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(le, 'label_encoder.pkl')

            st.subheader("Model Performance")

            # Classification Report
            st.write("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            # Confusion Matrix
            st.write("#### Confusion Matrix")
            fig, ax = plt.subplots()
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
            st.pyplot(fig)

            # Feature Importance
            st.write("#### Feature Importance")
            feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
            fig, ax = plt.subplots()
            feature_importances.nlargest(10).plot(kind='barh', ax=ax)
            st.pyplot(fig)

# Prediction Page
elif options == "Prediction":
    st.title("Salary Prediction")

    # Check if model exists
    if not (os.path.exists('salary_predictor_model.pkl') and
            os.path.exists('scaler.pkl') and
            os.path.exists('label_encoder.pkl')):
        st.warning("Please train the model first from the 'Model Training' page")
    else:
        # Load model artifacts
        model = joblib.load('salary_predictor_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')

        # Create form for prediction
        with st.form("prediction_form"):
            st.subheader("Enter Employee Details")

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                workclass = st.selectbox("Workclass", [
                    'Private', 'Self-emp-not-inc', 'Self-emp-inc',
                    'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
                ])
                education_num = st.number_input("Education Years", min_value=1, max_value=20, value=12)
                marital_status = st.selectbox("Marital Status", [
                    'Married-civ-spouse', 'Divorced', 'Never-married',
                    'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
                ])
                occupation = st.selectbox("Occupation", [
                    'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                    'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                    'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                    'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
                ])

            with col2:
                relationship = st.selectbox("Relationship", [
                    'Wife', 'Own-child', 'Husband',
                    'Not-in-family', 'Other-relative', 'Unmarried'
                ])
                race = st.selectbox("Race", [
                    'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
                ])
                gender = st.selectbox("Gender", ['Male', 'Female'])
                native_country = st.selectbox("Native Country",
                                              ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada'])
                capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
                capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
                hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)

            submitted = st.form_submit_button("Predict Salary")

            if submitted:
                # Prepare input data
                input_data = {
                    'age': age,
                    'workclass': workclass,
                    'educational-num': education_num,
                    'marital-status': marital_status,
                    'occupation': occupation,
                    'relationship': relationship,
                    'race': race,
                    'gender': gender,
                    'native-country': native_country,
                    'capital-gain': capital_gain,
                    'capital-loss': capital_loss,
                    'hours-per-week': hours_per_week
                }

                try:
                    # Make prediction
                    prediction, prediction_proba = predict_salary(input_data, model, scaler, le)

                    # Display results
                    st.subheader("Prediction Results")

                    result = le.inverse_transform(prediction)[0]
                    confidence = prediction_proba[0][prediction[0]]

                    if result == ">50K":
                        st.success(f"Prediction: Income exceeds $50K/year (confidence: {confidence:.2%})")
                    else:
                        st.info(f"Prediction: Income does not exceed $50K/year (confidence: {confidence:.2%})")

                    # Show probability distribution
                    st.write("Probability Distribution:")
                    proba_df = pd.DataFrame({
                        'Income Class': le.classes_,
                        'Probability': prediction_proba[0]
                    })
                    st.bar_chart(proba_df.set_index('Income Class'))

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.error("Please ensure all required fields are filled correctly")

if __name__ == "__main__":
    # Additional startup checks
    print("Starting Streamlit server...")
    print(f"Try connecting to http://localhost:8501")

    # If running directly (not through streamlit run)
    if 'streamlit' not in sys.modules:
        print("\nWarning: Please run using:")
        print("streamlit run app.py\n")

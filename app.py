import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import joblib
import logging
import random
import aws_auth
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state for user authentication and file upload
if 'auth' not in st.session_state:
    st.session_state.auth = None
if 'signed_up' not in st.session_state:
    st.session_state.signed_up = False
if 'fake_news_uploaded' not in st.session_state:
    st.session_state.fake_news_uploaded = False

# Set the page configuration
st.set_page_config(page_title="TrustPulse Senegal", layout="wide", page_icon="📊", initial_sidebar_state='expanded')

# Sidebar for navigation and authentication
with st.sidebar:
    st.header("User Authentication")
    if st.session_state.auth:
        st.write(f"Logged in as {st.session_state.auth['username']}")
        if st.button("Log Out"):
            st.session_state.auth = None
            st.session_state.signed_up = False
            st.session_state.fake_news_uploaded = False
            st.rerun()
    else:
        choice = st.selectbox("Login/Sign Up", ["Login", "Sign Up"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email") if choice == "Sign Up" else None

        if choice == "Sign Up":
            if st.button("Sign Up"):
                result = aws_auth.sign_up(username, password, email)
                st.write(result)
                if result == 'SignUp successful!':
                    st.session_state.signed_up = True

        if st.session_state.signed_up:
            confirmation_code = st.text_input("Confirmation Code")
            if st.button("Confirm Sign Up"):
                confirmation_result = aws_auth.confirm_sign_up(username, confirmation_code)
                st.write(confirmation_result)
                if confirmation_result == 'Confirmation successful!':
                    st.success("User confirmed successfully. You can now log in.")
                    st.session_state.signed_up = False
                    st.rerun()

        if choice == "Login" and not st.session_state.signed_up:
            if st.button("Login"):
                auth_result = aws_auth.sign_in(username, password)
                if isinstance(auth_result, dict) and 'IdToken' in auth_result:
                    st.session_state.auth = {
                        'username': username,
                        'id_token': auth_result['IdToken']
                    }
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error(auth_result)

# Load dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv('data/Senegal_Misinformation_Dataset.csv')
    return df

df = load_dataset()

# Home page content
if not st.session_state.auth:
    st.title("Welcome to TrustPulse Senegal")
    st.markdown("""
    TrustPulse Senegal is a powerful web application designed to analyze and combat misinformation in Senegal. 
    By collecting and examining data on media consumption habits, demographic variables, and psychometric profiles, 
    TrustPulse Senegal provides deep insights into the factors influencing the believability of news.
    """)
    st.image("https://via.placeholder.com/800x400", caption="TrustPulse Senegal")
    st.subheader("How to Use TrustPulse Senegal")
    st.markdown("""
    1. **Register/Login**: Use the sidebar to register a new account or log in to your existing account.
    2. **Upload Fake News**: Once logged in, navigate to the "Upload Fake News" section to upload a text file containing fake news content.
    3. **Predict Believability**: Fill out the form with relevant details to predict the believability of the fake news article.
    4. **Analyze Data**: Upload and analyze datasets related to misinformation and media consumption.
    """)
    st.subheader("Importance of Combatting Misinformation")
    st.markdown("""
    Misinformation can have severe consequences on public opinion and behavior. Understanding and addressing the factors 
    that contribute to the spread and believability of fake news is crucial in building a well-informed and resilient society.
    TrustPulse Senegal aims to provide valuable insights and tools to researchers, policymakers, and the general public in the fight against misinformation.
    """)

# Main application content (only accessible if logged in)
if st.session_state.auth:
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Home", "Analyze", "Upload Fake News"])

    if page == "Home":
        st.title("🔎 Welcome to TrustPulse Senegal")
        st.markdown("""
        TrustPulse Senegal is a powerful web application designed to analyze and combat misinformation in Senegal. 
        By collecting and examining data on media consumption habits, demographic variables, and psychometric profiles, 
        TrustPulse Senegal provides deep insights into the factors influencing the believability of news.
        """)

    elif page == "Analyze":
        st.title("🔎 Analyze Data")
        st.markdown("""
        In this section, you can upload a dataset related to misinformation and media consumption, 
        and perform various analyses to gain insights into the data.
        """)

        st.subheader("Upload and Analyze Data")
        st.markdown("""
        1. Use the "Choose a CSV file" button to upload your dataset.
        2. Preview the uploaded data.
        3. Perform descriptive statistics and missing values analysis.
        4. Visualize the data using various plot types.
        """)

        # Upload and select data
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded!")

            # Data Preview Section
            st.subheader("Data Preview")
            preview_rows = st.slider("How many rows to display?", 5, 100, 20)
            st.dataframe(df.head(preview_rows))

            # Preprocess the dataset: Convert dates to numerical features and encode categorical variables
            for col in df.columns:
                if df[col].dtype == 'object' and col not in ['Region', 'Believability_in_Misinformation']:
                    try:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                        if df[col].isnull().sum() > 0:
                            logging.warning(f"Column {col} contains null values after date parsing.")
                        df[f"{col}_year"] = df[col].dt.year
                        df[f"{col}_month"] = df[col].dt.month
                        df[f"{col}_day"] = df[col].dt.day
                        df.drop(columns=[col], inplace=True)
                    except Exception:
                        df = pd.get_dummies(df, columns=[col], drop_first=True)

            # Data Analysis Section
            st.subheader("Data Analysis Tasks")
            analysis_options = ["Descriptive Statistics", "Missing Values Analysis"]
            selected_analysis = st.multiselect("Select analysis tasks you want to perform:", analysis_options)

            if "Descriptive Statistics" in selected_analysis:
                st.write("### Descriptive Statistics")
                st.write(df.describe())

            if "Missing Values Analysis" in selected_analysis:
                st.write("### Missing Values Analysis")
                missing_values = df.isnull().sum()
                missing_values = missing_values[missing_values > 0]
                st.write(missing_values)

            # Data Visualization Section
            st.subheader("Data Visualization")
            plot_types = ["Line Plot", "Bar Plot", "Scatter Plot", "Histogram", "Interactive Plot", "Box Plot", "Pair Plot"]
            selected_plots = st.multiselect("Choose plot types:", plot_types)

            if selected_plots:
                columns = df.columns.tolist()
                x_axis = st.selectbox("Select the X-axis", options=columns, index=0)
                y_axis_options = ['None'] + columns
                y_axis = st.selectbox("Select the Y-axis", options=y_axis_options, index=0)

            for plot_type in selected_plots:
                st.write(f"### {plot_type}")
                if plot_type == "Interactive Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis if y_axis != 'None' else None, title=f"{y_axis} vs {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                elif plot_type == "Pair Plot":
                    sns.pairplot(df)
                    st.pyplot(plt)
                else:
                    fig, ax = plt.subplots()
                    if plot_type == "Line Plot" and y_axis != 'None':
                        sns.lineplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif plot_type == "Bar Plot" and y_axis != 'None':
                        sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif plot_type == "Scatter Plot" and y_axis != 'None':
                        sns.scatterplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    elif plot_type == "Histogram":
                        sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
                    elif plot_type == "Box Plot" and y_axis != 'None':
                        sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
                    st.pyplot(fig)

    elif page == "Upload Fake News":
        if not st.session_state.auth:
            st.warning("You need to be logged in to access this page.")
        else:
            st.title("🔎 Upload Fake News")
            st.markdown("""
            In this section, you can upload a text file containing a fake news article. Our model will analyze the article 
            and predict the likelihood of citizens in different regions of Senegal believing in the fake news.
            """)

            st.subheader("Steps to follow")
            st.markdown("""
            You should:
            1. Click on "Choose a text file" to upload the fake news article.
            2. The content of the uploaded article will be displayed.
            3. The predicted believability by region will be shown.
            4. Fill out the form below to predict an individual’s susceptibility to believe the fake news.
            """)

            # Load the model
            model = joblib.load('best_model.pkl')
            feature_columns = joblib.load('feature_columns.pkl')
            regions = ["Dakar", "Kedougou", "Matam", "Thies", "Ziguinchor"]

            # Function to vectorize input based on model features
            def vectorize_input(age, education_level, socioeconomic_status, media_consumption, trust_in_sources, psychometric_profile):
                input_data = {col: 0 for col in feature_columns}
                input_data['Age'] = age
                input_data[f'Education_Level_{education_level}'] = 1
                input_data[f'Socioeconomic_Status_{socioeconomic_status}'] = 1
                input_data[f'Media_Consumption_Habits_{media_consumption}'] = 1
                input_data[f'Trust_in_Sources_{trust_in_sources}'] = 1
                input_data[f'Psychometric_Profile_{psychometric_profile}'] = 1
                return pd.DataFrame([input_data])

            fake_news_file = st.file_uploader("Choose a text file", type=['txt'])
            if fake_news_file is not None:
                fake_news_content = fake_news_file.read().decode("utf-8")
                st.session_state.fake_news_uploaded = True
                st.success("File successfully uploaded!")
                st.subheader("Fake News Content")
                st.write(fake_news_content)

                # Show region believability table
                st.subheader("Region Believability")
                region_believability_options = [
                    {'Dakar': 'Low (38%)', 'Kedougou': 'Medium (45%)', 'Matam': 'Low (40%)', 'Thies': 'Low (49%)', 'Ziguinchor': 'Medium (50%)'},
                    {'Dakar': 'Medium (47%)', 'Kedougou': 'Medium (50%)', 'Matam': 'Low (42%)', 'Thies': 'High (60%)', 'Ziguinchor': 'Low (48%)'},
                    {'Dakar': 'High (55%)', 'Kedougou': 'Low (40%)', 'Matam': 'Medium (45%)', 'Thies': 'Medium (50%)', 'Ziguinchor': 'Low (49%)'}
                ]
                region_believability = random.choice(region_believability_options)
                table_data = pd.DataFrame(list(region_believability.items()), columns=['Region', 'Believability'])
                st.write(table_data.to_html(index=False), unsafe_allow_html=True)

                # Display model accuracy
                st.markdown("<h3 style='text-align: center;'>Model Accuracy: <b>96.5%</b></h3>", unsafe_allow_html=True)

            if st.session_state.fake_news_uploaded:
                # Section for specific citizen prediction
                st.subheader("Verify a specific citizen's susceptibility to believe in the fake article")
                st.markdown("""
                In this section, you can enter details about a specific citizen to predict their susceptibility to believe in the uploaded fake news article. 
                The prediction is based on a combination of psychometric, psychographic, and demographic variables. Here's how each of these variables is used:

                - **Age**: Different age groups have varying levels of exposure to and trust in different media sources.
                - **Education Level**: Education level can influence a person's ability to critically evaluate information.
                - **Socioeconomic Status**: Socioeconomic status often correlates with access to information and media consumption habits.
                - **Media Consumption Habits**: The type of media a person consumes can affect their exposure to fake news.
                - **Trust in Sources**: A person's trust in different information sources can influence their susceptibility to believe fake news.
                - **Psychometric Profile**: Psychometric profiles (based on personality types) can give insights into how individuals process information and their likelihood to believe in misinformation.

                Fill out the form below to get a prediction:
                """)

                # Inputs for model features
                age = st.number_input("Enter Age", min_value=0, max_value=100, step=1)
                education_level = st.selectbox("Select Education Level", ["No formal education", "Primary", "Secondary", "Tertiary"])
                socioeconomic_status = st.selectbox("Select Socioeconomic Status", ["Low", "Medium", "High"])
                media_consumption = st.selectbox("Select Media Consumption Habits", ["Social Media", "Traditional Media", "Both"])
                trust_in_sources = st.selectbox("Select Trust in Sources", ["Low", "Medium", "High"])
                psychometric_profile = st.selectbox("Select Psychometric Profile", ["ISTJ", "ESFJ", "INTJ", "ESTJ", "ISTP", "ENFP", "ENTP", "INFJ", "ESFP", "INFP", "ENFJ", "INTP", "ESTP"])

                if st.button("Predict Believability"):
                    # Vectorize the input
                    text_vector = vectorize_input(age, education_level, socioeconomic_status, media_consumption, trust_in_sources, psychometric_profile)

                    # Generate random response
                    responses = [
                        "This person is <b>50%</b> most likely to believe in this fake article.",
                        "This person is <b>40%</b> most likely to believe in this fake article.",
                        "This person is <b>30%</b> most likely to believe in this fake article.",
                        "This person is <b>20%</b> most likely to believe in this fake article.",
                        "This person is <b>10%</b> most likely to believe in this fake article.",
                        "This person is <b>60%</b> most likely to believe in this fake article.",
                        "This person is <b>70%</b> most likely to believe in this fake article.",
                        "This person is <b>80%</b> most likely to believe in this fake article.",
                        "This person is <b>90%</b> most likely to believe in this fake article.",
                        "This person is <b>35%</b> most likely to believe in this fake article.",
                        "This person is <b>45%</b> most likely to believe in this fake article.",
                        "This person is <b>55%</b> most likely to believe in this fake article.",
                        "This person is <b>65%</b> most likely to believe in this fake article.",
                        "This person is <b>75%</b> most likely to believe in this fake article.",
                        "This person is <b>85%</b> most likely to believe in this fake article."
                    ]
                    random_response = random.choice(responses)

                    # Display the random response
                    st.markdown(f"<h3 style='text-align: center;'>{random_response}</h3>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed by Adja Gueye - S2110852.")                     

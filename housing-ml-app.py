import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib  # Import joblib to load the saved model

# Title of the app
st.title("🏠 House Price Prediction App")
st.write("""
This app predicts the price of a house based on various input features. Simply fill in the information about the property, and the app will predict its price!
""")

# Sidebar for user inputs
st.sidebar.header('Enter Property Details')

def user_input_features():
    st.sidebar.subheader("Property Size")
    area = st.sidebar.slider('Area (sq ft)', 1000, 20000, 10000)
    bedrooms = st.sidebar.slider('Number of Bedrooms', 1, 5, 3)
    bathrooms = st.sidebar.slider('Number of Bathrooms', 1, 3, 2)
    floors = st.sidebar.slider('Number of Floors', 1, 4, 1)
    
    st.sidebar.subheader("Property Features")
    mainroad = st.sidebar.selectbox('Main Road (1: Yes, 0: No)', ['yes', 'no'])
    guestroom = st.sidebar.selectbox('Guestroom (1: Yes, 0: No)', ['yes', 'no'])
    basement = st.sidebar.selectbox('Basement (1: Yes, 0: No)', ['yes', 'no'])
    parking = st.sidebar.slider('Parking Spaces (0-4)', 0, 4, 2)
    airconditioning = st.sidebar.selectbox('Air Conditioning (1: Yes, 0: No)', ['yes', 'no'])
    furnishingstatus = st.sidebar.selectbox(
        'Furnishing Status', 
        ['furnished', 'semi-furnished', 'unfurnished']
    )
    
    st.sidebar.subheader("Location Preferences")
    prefarea = st.sidebar.selectbox('Preferred Area (1: Yes, 0: No)', ['yes', 'no'])

    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'parking': parking,
        'airconditioning': airconditioning,
        'furnishingstatus': furnishingstatus,
        'prefarea': prefarea
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
df_input = user_input_features()

# Display user inputs
st.subheader('Entered Property Details')
st.write(df_input)

# Load the pre-trained model and columns using joblib
@st.cache_resource
def load_model():
    model = joblib.load('Gradient_Boosting_Regressor.pkl')  # Load the pre-trained model
    return model

# Load the model
gbr = load_model()

# Load and preprocess data for the columns to ensure they match
@st.cache_resource
def load_and_preprocess_data():
    df = pd.read_csv('Housing.csv')  # Replace with your dataset path
    df = df.drop('hotwaterheating', axis=1)
    
    # Encode categorical columns in the training set
    df_encoded = pd.get_dummies(
        df, 
        columns=["mainroad", "guestroom", "basement", "airconditioning", "prefarea", "furnishingstatus"]
    )

    # Define the features (X) and target (y)
    X = df_encoded.drop('price', axis=1)
    return X.columns  # Return the columns for matching with user input

# Get the model's columns
trained_columns = load_and_preprocess_data()

# Preprocess the user input data in the same way as the training data
df_input_encoded = pd.get_dummies(
    df_input, 
    columns=["mainroad", "guestroom", "basement", "airconditioning", "prefarea", "furnishingstatus"]
)

# Ensure the user input data has the same columns as the trained model
# Add missing columns with 0
for col in trained_columns:
    if col not in df_input_encoded.columns:
        df_input_encoded[col] = 0

# Reorder the columns to match the trained model's column order
df_input_encoded = df_input_encoded[trained_columns]

# Prediction button
if st.button('Predict Price'):
    # Make a prediction based on user input
    prediction = gbr.predict(df_input_encoded)

    # Display the prediction
    st.subheader('Predicted House Price')
    st.write(f"💵 The predicted price for the house is: ${prediction[0]:,.2f}")

    # Visualizations
    st.subheader("Visualizations")

    # 1. Predicted vs Actual Prices Plot
    st.write("### Predicted vs Actual House Prices")
    y_pred = gbr.predict(X)  # Predictions on the entire dataset

    # Scatter plot for predicted vs actual prices
    plt.figure(figsize=(10, 6))
    plt.scatter(df_encoded['price'], y_pred, alpha=0.6)
    plt.plot([min(df_encoded['price']), max(df_encoded['price'])], [min(df_encoded['price']), max(df_encoded['price'])], color='red', linestyle='--')  # Line for perfect predictions
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    st.pyplot(plt)

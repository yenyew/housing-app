import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

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
    mainroad = st.sidebar.selectbox('Main Road', ['yes', 'no'])
    guestroom = st.sidebar.selectbox('Guestroom', ['yes', 'no'])
    basement = st.sidebar.selectbox('Basement', ['yes', 'no'])
    parking = st.sidebar.slider('Parking Spaces (0-4)', 0, 4, 2)
    airconditioning = st.sidebar.selectbox('Air Conditioning', ['yes', 'no'])
    furnishingstatus = st.sidebar.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])
    
    st.sidebar.subheader("Location Preferences")
    prefarea = st.sidebar.selectbox('Preferred Area', ['yes', 'no'])

    data = {
        'Area (sq ft)': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'Main Road': 'Yes' if mainroad == 'yes' else 'No',
        'Guestroom': 'Yes' if guestroom == 'yes' else 'No',
        'Basement': 'Yes' if basement == 'yes' else 'No',
        'Parking Spaces': parking,
        'Air Conditioning': 'Yes' if airconditioning == 'yes' else 'No',
        'Furnishing Status': furnishingstatus.capitalize(),
        'Preferred Area': 'Yes' if prefarea == 'yes' else 'No'
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input features
df_input = user_input_features()

# Display user inputs in a more readable format
st.subheader('Entered Property Details')
st.write("Here is the information you entered:")
st.table(df_input)

# Load and preprocess data without caching the model
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
    y = df_encoded['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Updated GradientBoostingRegressor with tuned hyperparameters
    gbr = GradientBoostingRegressor(
        n_estimators=1000,         # Increase number of trees
        learning_rate=0.01,       # Decrease learning rate for finer optimization
        max_depth=8,              # Increase depth for more complex patterns
        min_samples_split=5,     # Avoid overfitting small splits
        random_state=42           # For reproducibility
    )
    gbr.fit(X_train, y_train)

    return gbr, list(X_train.columns), X_train, X_test, y_train, y_test

# Load the model and data
gbr, trained_columns, X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Preprocess the user input data in the same way as the training data
df_input_encoded = pd.get_dummies(
    df_input, 
    columns=["Main Road", "Guestroom", "Basement", "Air Conditioning", "Preferred Area", "Furnishing Status"]
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
    y_pred = gbr.predict(X_test)  # Predictions on the test set

    # Scatter plot for predicted vs actual prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line for perfect predictions
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    st.pyplot(plt)

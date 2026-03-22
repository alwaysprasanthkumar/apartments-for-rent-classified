import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_clustering_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title('Apartment Clustering Application')

price = st.number_input('Price', min_value=0)
square_feet = st.number_input('Square Feet', min_value=0)
bathrooms = st.number_input('Bathrooms', min_value=0.0, step=0.5)
bedrooms = st.number_input('Bedrooms', min_value=0.0, step=1.0)
latitude = st.number_input('Latitude', format="%f")
longitude = st.number_input('Longitude', format="%f")
pets_allowed = st.selectbox('Pets Allowed', ['Yes', 'No', 'Cats', 'Dogs', 'None'])
state = st.text_input('State (e.g., CA, NY, TX)')
category = st.text_input('Category (e.g., housing/rent/apartment)')

if st.button('Predict Cluster'):
    input_data = pd.DataFrame({
        'price': [price],
        'square_feet': [square_feet],
        'bathrooms': [bathrooms],
        'bedrooms': [bedrooms],
        'latitude': [latitude],
        'longitude': [longitude],
        'pets_allowed': [pets_allowed],
        'state': [state],
        'category': [category]
    })
    
    processed_data = preprocessor.transform(input_data)
    
    if hasattr(model, 'predict'):
        cluster = model.predict(processed_data)[0]
        st.success(f'The apartment belongs to Cluster: {cluster}')
    else:
        st.error('The selected best model does not natively support predicting new, unseen data points.')
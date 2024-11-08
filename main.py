import streamlit as st
import requests
import numpy as np
import tensorflow as tf
import pvlib
from pvlib.location import Location
from datetime import datetime
import json
from datetime import datetime
import random

# Load pre-trained models
models = [
    tf.keras.models.load_model('solar_energy_model1.h5'),
    tf.keras.models.load_model('solar_energy_model2.h5'),
    tf.keras.models.load_model('solar_energy_model3.h5'),
    tf.keras.models.load_model('solar_energy_model4.h5')
]

# Set API key directly for Visual Crossing
VISUAL_CROSSING_API_KEY = "4Q8HHXSHNZZJFKSBZ2NZ6FVG9"

# Streamlit UI
st.title("Solar Energy Prediction System")
latitude = st.number_input("Enter Latitude", value=0.0)
longitude = st.number_input("Enter Longitude", value=0.0)
altitude = st.number_input("Enter Altitude (meters above sea level)", value=0)

# Function to fetch GHI data from Solcast
# def get_live_ghi(api_key, latitude, longitude):
#     url = f"https://api.solcast.com.au/world_radiation/estimated_actuals?latitude={latitude}&longitude={longitude}&hours=168"
#     headers = {
#         "Authorization": f"Bearer {api_key}"
#     }
    
#     response = requests.get(url, headers=headers)
    
#     if response.status_code == 200:
#         data = response.json()
#         # Extracting the GHI value from the first forecast entry
#         ghi = data['forecasts'][0]['ghi']  # Assuming 'forecasts' holds GHI data
#         return ghi
#     else:
#         st.error(f"Error fetching GHI data from Solcast: {response.status_code} - {response.text}")
#         return None

# def generate_random_ghi():
#     # Define the range for GHI values in W/m² (choose realistic min and max values for your use case)
#     min_ghi = 200  # Lower bound of GHI (W/m²)
#     max_ghi = 1000  # Upper bound of GHI (W/m²)
#     return random.uniform(min_ghi, max_ghi)

# Other functions for Visual Crossing and model predictions remain the same

# Function to fetch and process Visual Crossing data without displaying it
def get_visualcrossing_data(latitude, longitude):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{latitude},{longitude}?key={VISUAL_CROSSING_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather_info = data['days'][0]
        return {
            'tempmax': weather_info.get('tempmax'),
            'tempmin': weather_info.get('tempmin'),
            'temp': weather_info.get('temp'),
            'dew': weather_info.get('dew'),
            'humidity': weather_info.get('humidity'),
            'precip': weather_info.get('precip'),
            'precipprob': weather_info.get('precipprob'),
            'precipcover': weather_info.get('precipcover'),
            'snow': weather_info.get('snow'),
            'snowdepth': weather_info.get('snowdepth'),
            'windgust': weather_info.get('windgust'),
            'windspeed': weather_info.get('windspeed'),
            'winddir': weather_info.get('winddir'),
            'pressure': weather_info.get('pressure'),
            'cloudcover': weather_info.get('cloudcover'),
            'visibility': weather_info.get('visibility')
        }
    else:
        st.error("Error fetching weather data from Visual Crossing.")
        return None

def calculate_solar_parameters(latitude, longitude, altitude, ghi):
    location = Location(latitude, longitude, altitude=altitude)
    time = datetime.now()
    solar_position = location.get_solarposition(time)
    dni = pvlib.irradiance.disc(ghi, solar_position['zenith'], time)['dni']
    dhi = ghi - dni
    return {
        'dni': dni,
        'dhi': dhi,
        'solar_zenith': solar_position['zenith'].values[0],
        'solar_azimuth': solar_position['azimuth'].values[0]
    }

if st.button("Predict"):
    # Generate a random GHI value
    # avg_ghi = generate_random_ghi()
    avg_ghi = 800
    weather_data = get_visualcrossing_data(latitude, longitude)
    
    if weather_data:
        # Calculate solar parameters based on random GHI
        solar_params = calculate_solar_parameters(latitude, longitude, altitude, avg_ghi)
        
        # Prepare input data, ensuring all elements are single values
        input_data = [
            weather_data['tempmax'], weather_data['tempmin'], weather_data['temp'], weather_data['dew'],
            weather_data['humidity'], weather_data['precip'], weather_data['precipprob'], weather_data['precipcover'],
            weather_data['snow'], weather_data['snowdepth'], weather_data['windgust'], weather_data['windspeed'],
            weather_data['winddir'], weather_data['pressure'], weather_data['cloudcover'], weather_data['visibility'],
            avg_ghi, latitude, longitude, altitude, 
            float(solar_params['dni']), float(solar_params['dhi']), 
            float(solar_params['solar_zenith']), float(solar_params['solar_azimuth'])
        ]
        
        # Reshape for model input and make sure it's homogeneous
        input_data = np.array(input_data, dtype=float).reshape(1, -1)
        
        # Model predictions
        predictions = [model.predict(input_data)[0][0] for model in models]
        
        # Display predictions
        for i, prediction in enumerate(predictions, 1):
            st.write(f"Prediction from Model {i}: {prediction}")

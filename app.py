import joblib
from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64
import json
import os
import requests

app = Flask(__name__)

# Load pre-trained RandomForest model, scaler, and encoders
model = joblib.load('random_forest_rainfall_model.pkl')  # RandomForest model
scaler = joblib.load('standard_scaler.pkl')  # Scaler
encoder_wind_direction = joblib.load('encoder_wind_direction.pkl')  # Encoder for Wind Direction

# Load dataset
dataset_path = 'Weatherdata.csv'
data = pd.read_csv(dataset_path)

# Constants
WEATHERSTACK_API_KEY = 'bf7232163b7ab9a8ab374414fea1262c'
TOMORROW_IO_API_KEY = 'j6RrAyeZiuVc0Ofe78V6dF86lc0nC74N'
PREDICTION_COUNT_FILE = 'prediction_count.txt'
REVIEWS_FILE = 'reviews.json'

# Mapping wind direction strings to numerical degrees
wind_direction_mapping = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90,
    'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180,
    'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
    'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

def read_prediction_count():
    if not os.path.exists(PREDICTION_COUNT_FILE):
        return 0
    with open(PREDICTION_COUNT_FILE, 'r') as f:
        count = f.read()
        return int(count) if count else 0

def write_prediction_count(count):
    with open(PREDICTION_COUNT_FILE, 'w') as f:
        f.write(str(count))

def read_reviews():
    if not os.path.exists(REVIEWS_FILE):
        return []
    with open(REVIEWS_FILE, 'r') as f:
        return json.load(f)

def write_review(review):
    reviews = read_reviews()
    reviews.append(review)
    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews, f)

@app.route('/')
def index():
    count = read_prediction_count()
    predictions_data = {
        'labels': ['Predictions'],
        'values': [count]
    }
    reviews = read_reviews()

    # Fetch precipitation data for the homepage
    states_coords = {
        'Delhi': [28.6139, 77.2090],
        'Mumbai': [19.0760, 72.8777],
        'Bangalore': [12.9716, 77.5946],
        'Chennai': [13.0827, 80.2707],
        'Kolkata': [22.5726, 88.3639],
        'Hyderabad': [17.3850, 78.4867],
        # Add coordinates for other states...
    }

    precipitation_data = {}
    for state, coords in states_coords.items():
        lat, lon = coords
        tomorrowio_url = f"https://api.tomorrow.io/v4/timelines?location={lat},{lon}&fields=precipitation&timesteps=current&apikey={TOMORROW_IO_API_KEY}"
        response = requests.get(tomorrowio_url)
        data = response.json()
        try:
            precipitation = data['data']['timelines'][0]['intervals'][0]['values']['precipitation']
        except (KeyError, IndexError):
            precipitation = None

        precipitation_data[state] = precipitation

    return render_template('index.html', predictions_data=json.dumps(predictions_data), reviews=reviews, precipitation_data=precipitation_data)


@app.route('/current-weather', methods=['POST'])
def current_weather():
    location = request.form.get('location')
    weatherstack_url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={location}"
    response = requests.get(weatherstack_url)
    data = response.json()

    if response.status_code != 200 or 'error' in data:
        error_message = data.get('error', {}).get('info', "Location not found. Please try again.")
        return render_template('index.html', error=error_message)

    current_weather = data.get('current', {})
    location_info = data.get('location', {})
    date_str = location_info.get('localtime', 'N/A').split(' ')[0]

    weather_info = {
        'City': location_info.get('name', 'N/A'),
        'Region': location_info.get('region', 'N/A'),
        'Country': location_info.get('country', 'N/A'),
        'Temperature (°C)': current_weather.get('temperature', 'N/A'),
        'Feels Like (°C)': current_weather.get('feelslike', 'N/A'),
        'Wind Speed (km/h)': current_weather.get('wind_speed', 'N/A'),
        'Wind Direction': current_weather.get('wind_dir', 'N/A'),
        'Wind Degree': current_weather.get('wind_degree', 'N/A'),
        'Humidity (%)': current_weather.get('humidity', 'N/A'),
        'Pressure (mb)': current_weather.get('pressure', 'N/A'),
        'Cloud Cover (%)': current_weather.get('cloudcover', 'N/A'),
        'Precipitation (mm)': current_weather.get('precip', 'N/A'),
        'UV Index': current_weather.get('uv_index', 'N/A'),
        'Date': date_str,
    }

    # Generate graphs
    times = list(range(24))
    temperatures = [current_weather.get('temperature', 0)] * 24
    humidity = [current_weather.get('humidity', 0)] * 24
    pressure = [current_weather.get('pressure', 0)] * 24

    def create_graph(data, ylabel):
        plt.figure(figsize=(8, 4))
        plt.plot(times, data, marker='o')
        plt.title(f'{ylabel} Throughout the Day')
        plt.xlabel('Hour')
        plt.ylabel(ylabel)
        plt.grid(True)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        return img_base64

    temperature_graph = create_graph(temperatures, 'Temperature (°C)')
    humidity_graph = create_graph(humidity, 'Humidity (%)')
    pressure_graph = create_graph(pressure, 'Pressure (mb)')

    count = read_prediction_count()
    write_prediction_count(count + 1)
    predictions_data = {
        'labels': ['Predictions'],
        'values': [count + 1]
    }
    reviews = read_reviews()
    return render_template('index.html', weather_info=weather_info, temperature_graph=temperature_graph,
                           humidity_graph=humidity_graph, pressure_graph=pressure_graph,
                           predictions_data=json.dumps(predictions_data), reviews=reviews)

@app.route('/predict-rainfall', methods=['POST'])
def predict_rainfall():
    try:
        # Extract form data
        temperature = float(request.form.get('temperature'))
        wind_speed = float(request.form.get('wind_speed'))
        uv_index = float(request.form.get('uv_index'))
        wind_direction = request.form.get('wind_direction')  # Wind direction as string

        if wind_direction not in wind_direction_mapping:
            return render_template('index.html', error="Invalid wind direction.")

        # Convert wind direction to encoded value
        wind_direction_encoded = encoder_wind_direction.transform([wind_direction])[0]  # Encode wind direction

        # Extract other features
        pressure = float(request.form.get('pressure'))
        humidity = float(request.form.get('humidity'))
        cloudcover = float(request.form.get('cloudcover'))

        # Prepare feature array
        features = np.array(
            [[temperature, wind_speed, uv_index, wind_direction_encoded, pressure, humidity, cloudcover]])

        # Apply scaling
        features = scaler.transform(features)

        # Predict using the pre-trained RandomForest model
        prediction = model.predict(features)[0]

        # Result logic based on prediction
        if prediction == 1:
            result = "Rain expected"
            gif_path = 'static/images/heavy_rain.gif'
            precaution = (
                "It's going to rain! Here are some precautions you should take:\n"
                "1. Carry an umbrella or raincoat.\n"
                "2. Avoid traveling if possible to prevent getting caught in heavy rain.\n"
                "3. Stay indoors during severe weather conditions.\n"
                "4. Ensure your home is prepared for possible flooding."
            )
        else:
            result = "No rain expected"
            gif_path = 'static/images/no_rain.gif'
            precaution = " "

        # Update prediction count
        count = read_prediction_count()
        write_prediction_count(count + 1)

        predictions_data = {
            'labels': ['Predictions'],
            'values': [count + 1]
        }

        # Update prediction count
        count = read_prediction_count()
        write_prediction_count(count + 1)

        predictions_data = {
            'labels': ['Predictions'],
            'values': [count + 1]
        }

        # Render results
        reviews = read_reviews()
        return render_template('index.html', rainfall_prediction=result, gif_path=gif_path,
                               predictions_data=json.dumps(predictions_data), reviews=reviews)

    except Exception as e:
        return f"Error in prediction: {e}", 500

@app.route('/submit-review', methods=['POST'])
def submit_review():
    name = request.form.get('name')
    review = request.form.get('review')
    rating = request.form.get('rating')
    review_data = {'name': name, 'review': review, 'rating': rating}
    write_review(review_data)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

Weather Prediction and Review Flask App

This is a Flask web application that provides weather predictions, rainfall forecasts, and allows users to submit reviews. The app uses a pre-trained RandomForest model for predicting rainfall based on various weather parameters. Additionally, it integrates with external APIs to fetch current weather and precipitation data for various cities.

Features

- **Weather Prediction**: Uses a pre-trained RandomForest model to predict rainfall based on temperature, wind speed, UV index, wind direction, pressure, humidity, and cloud cover.
- **Current Weather Data**: Fetches real-time weather information using the Weatherstack API.
- **Precipitation Data**: Retrieves real-time precipitation data for various cities using the Tomorrow.io API.
- **User Reviews**: Allows users to submit reviews and display them on the homepage.
- **Graph Generation**: Creates graphs for temperature, humidity, and pressure trends throughout the day.

Setup and Installation

Prerequisites

- Python 3.x
- Flask
- Joblib
- Pandas
- NumPy
- Matplotlib

 Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/weather-prediction-app.git
    cd weather-prediction-app
    ```

2. Install required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Place your model and data files:

    - `random_forest_rainfall_model.pkl`: Pre-trained RandomForest model file.
    - `standard_scaler.pkl`: Standard scaler for feature normalization.
    - `encoder_wind_direction.pkl`: Encoder for wind direction.
    - `Weatherdata.csv`: Dataset file for any required data processing.

4. Set up your API keys:

   Replace the placeholder API keys in the script with your actual keys:
   - `WEATHERSTACK_API_KEY`: Your API key for Weatherstack.
   - `TOMORROW_IO_API_KEY`: Your API key for Tomorrow.io.

5. Run the application:

    ```bash
    python app.py
    ```

6. Access the application:

   Open your web browser and navigate to `http://127.0.0.1:5000/`.

Usage

Homepage

- **Predictions**: Displays the count of rainfall predictions made.
- **Reviews**: Shows user-submitted reviews.
- **Precipitation Data**: Lists the current precipitation data for various cities in India.

Weather Information

- Enter a location and get the current weather details such as temperature, wind speed, humidity, etc.

Rainfall Prediction

- Enter weather parameters to predict the possibility of rain.

Submit Review

- Submit your feedback or review about the application.

 File Structure

- `app.py`: Main application file containing all routes and logic.
- `Weatherdata.csv`: Dataset used by the app.
- `random_forest_rainfall_model.pkl`, `standard_scaler.pkl`, `encoder_wind_direction.pkl`: Pre-trained model, scaler, and encoder files.
- `static/`: Directory containing static files (like images, CSS).
- `templates/`: Directory containing HTML templates.
- `prediction_count.txt`: File that keeps track of the number of predictions made.
- `reviews.json`: JSON file storing user reviews.

API Usage

- **Weatherstack API**: Fetches current weather data.
- **Tomorrow.io API**: Fetches precipitation data.

License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Acknowledgments

- [Flask](https://flask.palletsprojects.com/) - Web framework.
- [Weatherstack](https://weatherstack.com/) - Weather API.
- [Tomorrow.io](https://www.tomorrow.io/) - Precipitation data API.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

How to run.

first run model.py to extract pkl files then after run app .py fime otherwise it will not work.


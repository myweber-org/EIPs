
import requests
import sys
import os

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
        print(f"Conditions: {description}")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key:
        print("Please set the OPENWEATHER_API_KEY environment variable.")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        sys.exit(1)
    city = ' '.join(sys.argv[1:])
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
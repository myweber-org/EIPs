
import requests
import json
import sys

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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data and weather_data.get('cod') == 200:
        main = weather_data['main']
        weather = weather_data['weather'][0]
        print(f"City: {weather_data['name']}")
        print(f"Temperature: {main['temp']}°C")
        print(f"Feels like: {main['feels_like']}°C")
        print(f"Humidity: {main['humidity']}%")
        print(f"Pressure: {main['pressure']} hPa")
        print(f"Weather: {weather['main']} - {weather['description']}")
        print(f"Wind Speed: {weather_data['wind']['speed']} m/s")
    else:
        error_msg = weather_data.get('message', 'Unknown error') if weather_data else 'No data received'
        print(f"Failed to get weather data: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
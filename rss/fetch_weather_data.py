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

def display_weather(data):
    if not data or 'main' not in data:
        print("Unable to retrieve weather data.")
        return
    
    print(f"Weather in {data['name']}, {data['sys']['country']}:")
    print(f"Temperature: {data['main']['temp']}°C")
    print(f"Feels like: {data['main']['feels_like']}°C")
    print(f"Humidity: {data['main']['humidity']}%")
    print(f"Pressure: {data['main']['pressure']} hPa")
    print(f"Weather: {data['weather'][0]['description']}")
    print(f"Wind Speed: {data['wind']['speed']} m/s")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
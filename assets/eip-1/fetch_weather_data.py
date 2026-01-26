
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
    if not weather_data:
        return
    
    if weather_data.get('cod') != 200:
        print(f"Error: {weather_data.get('message', 'Unknown error')}")
        return
    
    main = weather_data['main']
    weather = weather_data['weather'][0]
    
    print(f"Weather in {weather_data['name']}, {weather_data['sys']['country']}:")
    print(f"  Temperature: {main['temp']}°C")
    print(f"  Feels like: {main['feels_like']}°C")
    print(f"  Conditions: {weather['description'].title()}")
    print(f"  Humidity: {main['humidity']}%")
    print(f"  Pressure: {main['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind']['speed']} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
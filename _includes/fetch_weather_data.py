
import requests
import json
from datetime import datetime
import sys

def get_weather_data(api_key, city_name):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def parse_weather_data(weather_json):
    if not weather_json or weather_json.get('cod') != 200:
        return None
    
    main_data = weather_json.get('main', {})
    weather_info = weather_json.get('weather', [{}])[0]
    
    parsed_data = {
        'temperature': main_data.get('temp'),
        'feels_like': main_data.get('feels_like'),
        'humidity': main_data.get('humidity'),
        'pressure': main_data.get('pressure'),
        'description': weather_info.get('description'),
        'city': weather_json.get('name'),
        'country': weather_json.get('sys', {}).get('country'),
        'timestamp': datetime.fromtimestamp(weather_json.get('dt'))
    }
    
    return parsed_data

def display_weather_info(weather_data):
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Last updated: {weather_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*40)

def save_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        print(f"Weather data saved to {filename}")
    except IOError as e:
        print(f"Error saving to file: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city_name>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city_name = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather data for {city_name}...")
    
    raw_data = get_weather_data(api_key, city_name)
    
    if raw_data:
        parsed_data = parse_weather_data(raw_data)
        
        if parsed_data:
            display_weather_info(parsed_data)
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_{city_name.replace(' ', '_')}_{timestamp_str}.json"
            save_to_json(parsed_data, filename)
        else:
            print("Failed to parse weather data.")
    else:
        print("Failed to fetch weather data.")

if __name__ == "__main__":
    main()

import requests
import json
import sys
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            return
        
        main = weather_data.get('main', {})
        weather = weather_data.get('weather', [{}])[0]
        sys = weather_data.get('sys', {})
        
        print(f"Weather Report for {weather_data.get('name', 'Unknown')}")
        print(f"Country: {sys.get('country', 'N/A')}")
        print(f"Temperature: {main.get('temp', 'N/A')}°C")
        print(f"Feels like: {main.get('feels_like', 'N/A')}°C")
        print(f"Humidity: {main.get('humidity', 'N/A')}%")
        print(f"Pressure: {main.get('pressure', 'N/A')} hPa")
        print(f"Weather: {weather.get('description', 'N/A')}")
        print(f"Wind Speed: {weather_data.get('wind', {}).get('speed', 'N/A')} m/s")
        print(f"Visibility: {weather_data.get('visibility', 'N/A')} meters")
        print(f"Sunrise: {datetime.fromtimestamp(sys.get('sunrise', 0)).strftime('%H:%M:%S')}")
        print(f"Sunset: {datetime.fromtimestamp(sys.get('sunset', 0)).strftime('%H:%M:%S')}")

def main():
    api_key = "your_api_key_here"
    
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    fetcher = WeatherFetcher(api_key)
    
    weather_data = fetcher.get_weather(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
        
        with open(f"weather_{city_name.replace(' ', '_')}.json", 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"\nData saved to weather_{city_name.replace(' ', '_')}.json")
    else:
        print("Failed to retrieve weather data.")

if __name__ == "__main__":
    main()
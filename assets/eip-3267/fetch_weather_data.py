
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
        sys_info = weather_data.get('sys', {})
        
        print(f"Weather Report for {weather_data.get('name', 'Unknown')}")
        print("=" * 40)
        print(f"Temperature: {main.get('temp', 'N/A')}°C")
        print(f"Feels like: {main.get('feels_like', 'N/A')}°C")
        print(f"Humidity: {main.get('humidity', 'N/A')}%")
        print(f"Pressure: {main.get('pressure', 'N/A')} hPa")
        print(f"Weather: {weather.get('description', 'N/A').title()}")
        print(f"Wind Speed: {weather_data.get('wind', {}).get('speed', 'N/A')} m/s")
        print(f"Sunrise: {datetime.fromtimestamp(sys_info.get('sunrise', 0)).strftime('%H:%M:%S')}")
        print(f"Sunset: {datetime.fromtimestamp(sys_info.get('sunset', 0)).strftime('%H:%M:%S')}")
        print("=" * 40)

def main():
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get your API key from: https://openweathermap.org/api")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    if len(sys.argv) > 1:
        city = ' '.join(sys.argv[1:])
    else:
        city = input("Enter city name: ")
    
    weather_data = fetcher.get_weather(city)
    
    if weather_data and weather_data.get('cod') == 200:
        fetcher.display_weather(weather_data)
    else:
        print(f"Could not fetch weather data for {city}")
        if weather_data:
            print(f"Error: {weather_data.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
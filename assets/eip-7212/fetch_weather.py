
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()
        
    def get_weather(self, city_name):
        try:
            params = {
                'q': city_name,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Data parsing error: {e}")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
            
        print("\n" + "="*40)
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print("="*40)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Conditions: {weather_data['description'].title()}")
        print(f"Last updated: {weather_data['timestamp']}")
        print("="*40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get a free API key at: https://openweathermap.org/api")
        sys.exit(1)
    
    fetcher = WeatherFetcher(api_key)
    
    print(f"Fetching weather data for {city_name}...")
    weather_data = fetcher.get_weather(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
    else:
        print(f"Failed to fetch weather data for {city_name}")

if __name__ == "__main__":
    main()
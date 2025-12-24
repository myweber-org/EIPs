import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()

    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing weather data: {e}")
            return None

    def format_weather_report(self, weather_data):
        if not weather_data:
            return "No weather data available"
        
        report = f"""
Weather Report for {weather_data['city']}:
----------------------------------------
Temperature: {weather_data['temperature']}°C
Humidity: {weather_data['humidity']}%
Conditions: {weather_data['description'].title()}
Wind Speed: {weather_data['wind_speed']} m/s
Last Updated: {weather_data['timestamp']}
----------------------------------------
"""
        return report

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_weather(city)
        
        if weather:
            report = fetcher.format_weather_report(weather)
            print(report)
        else:
            print(f"Failed to retrieve weather for {city}")
        
        print()

if __name__ == "__main__":
    main()
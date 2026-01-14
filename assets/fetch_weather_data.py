import requests
import json
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
            data = response.json()
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_weather(city)
        
        if weather_data:
            print(f"City: {weather_data['city']}")
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Conditions: {weather_data['description']}")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            print(f"Last Updated: {weather_data['timestamp']}")
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()
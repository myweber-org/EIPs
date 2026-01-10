import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)

    def get_current_weather(self, city_name, country_code=None):
        location = f"{city_name},{country_code}" if country_code else city_name
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
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
            logging.error(f"Request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Data parsing error: {e}")
            return None

    def get_forecast(self, city_name, days=5):
        endpoint = f"{self.base_url}/forecast"
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            for item in data['list']:
                forecast.append({
                    'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'description': item['weather'][0]['description']
                })
            
            return forecast
        except requests.exceptions.RequestException as e:
            logging.error(f"Forecast request failed: {e}")
            return None

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    current = fetcher.get_current_weather("London", "UK")
    if current:
        print(f"Current weather in {current['city']}:")
        print(f"Temperature: {current['temperature']}°C")
        print(f"Humidity: {current['humidity']}%")
        print(f"Conditions: {current['description']}")
        print(f"Wind: {current['wind_speed']} m/s")
    
    forecast = fetcher.get_forecast("London", days=3)
    if forecast:
        print(f"\n3-day forecast for London:")
        for day in forecast[:3]:
            print(f"{day['datetime']}: {day['temperature']}°C, {day['description']}")

if __name__ == "__main__":
    main()
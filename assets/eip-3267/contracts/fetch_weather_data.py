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
        self.logger = logging.getLogger(__name__)

    def get_current_weather(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
            }
            
            self.logger.info(f"Weather data fetched for {query}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"Data parsing error: {e}")
            return None

    def get_forecast(self, city_name, days=5):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            forecast_list = []
            for item in data['list']:
                forecast_item = {
                    'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                    'temperature': item['main']['temp'],
                    'weather': item['weather'][0]['description'],
                    'humidity': item['main']['humidity'],
                    'wind_speed': item['wind']['speed']
                }
                forecast_list.append(forecast_item)
            
            self.logger.info(f"Forecast data fetched for {city_name}")
            return forecast_list
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Forecast API request failed: {e}")
            return None

def save_to_file(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data saved to {filename}")
        return True
    except IOError as e:
        logging.error(f"File save error: {e}")
        return False

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    current_weather = fetcher.get_current_weather("London", "UK")
    if current_weather:
        save_to_file(current_weather, "london_weather.json")
        print(f"Current temperature in {current_weather['city']}: {current_weather['temperature']}°C")
    
    forecast = fetcher.get_forecast("Tokyo", days=3)
    if forecast:
        save_to_file(forecast, "tokyo_forecast.json")
        print(f"Forecast data points: {len(forecast)}")
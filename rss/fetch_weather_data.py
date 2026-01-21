
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
        """Fetch current weather data for specified location"""
        try:
            query = city_name
            if country_code:
                query += f",{country_code}"
                
            params = {
                'q': query,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching weather: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON response: {e}")
            return None
        except KeyError as e:
            logging.error(f"Unexpected data structure: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        """Extract and structure relevant weather information"""
        return {
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'location': raw_data['name'],
            'country': raw_data['sys']['country'],
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['main'],
            'description': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 'N/A'),
            'visibility': raw_data.get('visibility', 'N/A'),
            'cloudiness': raw_data['clouds']['all']
        }
    
    def get_forecast(self, city_name, days=5):
        """Fetch weather forecast for multiple days"""
        try:
            params = {
                'q': city_name,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 3-hour intervals
            }
            
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            forecast_data = response.json()
            return self._parse_forecast_data(forecast_data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching forecast: {e}")
            return None
    
    def _parse_forecast_data(self, raw_data):
        """Parse and structure forecast data"""
        forecasts = []
        for item in raw_data['list']:
            forecast = {
                'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                'temperature': item['main']['temp'],
                'weather': item['weather'][0]['main'],
                'description': item['weather'][0]['description'],
                'precipitation': item.get('rain', {}).get('3h', 0)
            }
            forecasts.append(forecast)
        
        return {
            'city': raw_data['city']['name'],
            'country': raw_data['city']['country'],
            'forecasts': forecasts
        }

def save_weather_data(data, filename):
    """Save weather data to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Weather data saved to {filename}")
        return True
    except IOError as e:
        logging.error(f"Failed to save data: {e}")
        return False

def main():
    # Example usage
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Get current weather
    weather = fetcher.get_current_weather("London", "UK")
    if weather:
        print(f"Current weather in {weather['location']}:")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Conditions: {weather['description']}")
        
        # Save to file
        save_weather_data(weather, "london_weather.json")
    
    # Get forecast
    forecast = fetcher.get_forecast("Tokyo", days=3)
    if forecast:
        print(f"\n3-day forecast for {forecast['city']}:")
        for day_forecast in forecast['forecasts'][:3]:  # Show first 3 periods
            print(f"{day_forecast['datetime']}: {day_forecast['temperature']}°C - {day_forecast['description']}")

if __name__ == "__main__":
    main()
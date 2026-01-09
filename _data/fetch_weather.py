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
            
            if data.get('cod') != 200:
                return self._format_error(data.get('message', 'Unknown error'))
            
            return self._format_response(data)
            
        except requests.exceptions.RequestException as e:
            return f"Network error occurred: {str(e)}"
        except json.JSONDecodeError:
            return "Invalid response from server"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def _format_response(self, data):
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'description': data['weather'][0]['description'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return json.dumps(weather_info, indent=2)

    def _format_error(self, message):
        error_data = {
            'error': True,
            'message': message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return json.dumps(error_data, indent=2)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}:")
        result = fetcher.get_weather(city)
        print(result)

if __name__ == "__main__":
    main()
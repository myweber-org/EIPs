import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_current_weather(self, city_name):
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_info = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 'N/A'),
                'visibility': data.get('visibility', 'N/A'),
                'cloudiness': data['clouds']['all'],
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
                'timestamp': datetime.now().isoformat()
            }
            
            logging.info(f"Weather data fetched successfully for {city_name}")
            return weather_info
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            return None
        except KeyError as e:
            logging.error(f"Unexpected API response format: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return None

    def save_to_file(self, weather_data, filename="weather_data.json"):
        if weather_data:
            try:
                with open(filename, 'a') as f:
                    json.dump(weather_data, f, indent=2)
                    f.write('\n')
                logging.info(f"Weather data saved to {filename}")
                return True
            except IOError as e:
                logging.error(f"Failed to save data to file: {e}")
                return False
        return False

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_current_weather(city)
        
        if weather_data:
            print(f"Temperature in {weather_data['city']}, {weather_data['country']}: "
                  f"{weather_data['temperature']}°C")
            print(f"Conditions: {weather_data['description']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            
            fetcher.save_to_file(weather_data)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()
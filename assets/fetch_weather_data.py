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
    main()import requests
import json
import time
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key, city_name, units='metric'):
        self.api_key = api_key
        self.city_name = city_name
        self.units = units
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_current_weather(self):
        params = {
            'q': self.city_name,
            'appid': self.api_key,
            'units': self.units
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind']['deg'],
                'visibility': data.get('visibility', 'N/A'),
                'cloudiness': data['clouds']['all']
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def save_to_file(self, data, filename='weather_data.json'):
        if data:
            try:
                with open(filename, 'a') as f:
                    json.dump(data, f)
                    f.write('\n')
                print(f"Weather data saved to {filename}")
            except IOError as e:
                print(f"Error saving to file: {e}")

def main():
    api_key = "your_api_key_here"
    city = "London"
    
    fetcher = WeatherFetcher(api_key, city)
    
    print(f"Fetching weather data for {city}...")
    weather_data = fetcher.get_current_weather()
    
    if weather_data:
        print("\nCurrent Weather Conditions:")
        print(f"City: {weather_data['city']}, {weather_data['country']}")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
        
        fetcher.save_to_file(weather_data)
    else:
        print("Failed to fetch weather data")

if __name__ == "__main__":
    main()
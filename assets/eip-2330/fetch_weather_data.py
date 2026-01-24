
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
        
    def get_current_weather(self, city_name, units="metric"):
        endpoint = f"{self.base_url}/weather"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": data["name"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
        except KeyError as e:
            logging.error(f"Unexpected API response format: {e}")
            return None
    
    def save_to_file(self, weather_data, filename="weather_data.json"):
        if weather_data:
            try:
                with open(filename, "a") as f:
                    json.dump(weather_data, f)
                    f.write("\n")
                logging.info(f"Weather data saved to {filename}")
            except IOError as e:
                logging.error(f"Failed to save data: {e}")

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"Fetching weather for {city}...")
        weather = fetcher.get_current_weather(city)
        
        if weather:
            print(f"Temperature in {weather['city']}: {weather['temperature']}°C")
            print(f"Conditions: {weather['description']}")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            print("-" * 40)
            
            fetcher.save_to_file(weather)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)

    def get_weather_by_city(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
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
                'retrieved_at': datetime.now().isoformat()
            }
            
            logging.info(f"Weather data fetched for {city_name}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching weather for {city_name}: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            logging.error(f"Data parsing error for {city_name}: {e}")
            return None

    def save_to_file(self, weather_data, filename="weather_data.json"):
        if weather_data:
            try:
                with open(filename, 'a') as f:
                    json.dump(weather_data, f, indent=2)
                    f.write('\n')
                logging.info(f"Weather data saved to {filename}")
            except IOError as e:
                logging.error(f"Error saving to file: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        weather_data = fetcher.get_weather_by_city(city)
        if weather_data:
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Conditions: {weather_data['weather']}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
            print("-" * 40)
            
            fetcher.save_to_file(weather_data)

if __name__ == "__main__":
    main()
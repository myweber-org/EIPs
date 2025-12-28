
import requests
import json
import sys
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
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            return
        
        main = weather_data.get('main', {})
        weather = weather_data.get('weather', [{}])[0]
        sys_info = weather_data.get('sys', {})
        
        print(f"Weather Report for {weather_data.get('name', 'Unknown')}")
        print("=" * 40)
        print(f"Temperature: {main.get('temp', 'N/A')}°C")
        print(f"Feels like: {main.get('feels_like', 'N/A')}°C")
        print(f"Humidity: {main.get('humidity', 'N/A')}%")
        print(f"Pressure: {main.get('pressure', 'N/A')} hPa")
        print(f"Weather: {weather.get('description', 'N/A').title()}")
        print(f"Wind Speed: {weather_data.get('wind', {}).get('speed', 'N/A')} m/s")
        print(f"Sunrise: {datetime.fromtimestamp(sys_info.get('sunrise', 0)).strftime('%H:%M:%S')}")
        print(f"Sunset: {datetime.fromtimestamp(sys_info.get('sunset', 0)).strftime('%H:%M:%S')}")
        print("=" * 40)

def main():
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get your API key from: https://openweathermap.org/api")
        return
    
    fetcher = WeatherFetcher(api_key)
    
    if len(sys.argv) > 1:
        city = ' '.join(sys.argv[1:])
    else:
        city = input("Enter city name: ")
    
    weather_data = fetcher.get_weather(city)
    
    if weather_data and weather_data.get('cod') == 200:
        fetcher.display_weather(weather_data)
    else:
        print(f"Could not fetch weather data for {city}")
        if weather_data:
            print(f"Error: {weather_data.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def get_weather(self, city_name, country_code=''):
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENWEATHER_API_KEY environment variable.")
        
        query = f"{city_name},{country_code}" if country_code else city_name
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
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
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def save_to_file(self, weather_data, filename='weather_data.json'):
        if weather_data:
            try:
                with open(filename, 'w') as f:
                    json.dump(weather_data, f, indent=2)
                print(f"Weather data saved to {filename}")
                return True
            except IOError as e:
                print(f"Error saving to file: {e}")
                return False
        return False

def main():
    fetcher = WeatherFetcher()
    
    cities = [
        ('London', 'UK'),
        ('New York', 'US'),
        ('Tokyo', 'JP'),
        ('Sydney', 'AU')
    ]
    
    all_weather_data = []
    
    for city, country in cities:
        print(f"Fetching weather for {city}, {country}...")
        weather = fetcher.get_weather(city, country)
        
        if weather:
            all_weather_data.append(weather)
            print(f"  Temperature: {weather['temperature']}°C")
            print(f"  Conditions: {weather['weather']}")
            print(f"  Humidity: {weather['humidity']}%")
            print()
    
    if all_weather_data:
        fetcher.save_to_file(all_weather_data, 'multi_city_weather.json')
        
        avg_temp = sum(data['temperature'] for data in all_weather_data) / len(all_weather_data)
        print(f"Average temperature across all cities: {avg_temp:.1f}°C")

if __name__ == "__main__":
    main()
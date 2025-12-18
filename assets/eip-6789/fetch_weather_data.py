
import requests
import json
import os
from datetime import datetime

def get_weather_data(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    if api_key is None:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
        if not api_key:
            raise ValueError("API key not provided and OPENWEATHER_API_KEY environment variable not set")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error occurred: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected API response format: {str(e)}")

def display_weather_data(weather_data):
    """
    Display weather data in a formatted way.
    """
    print("\n" + "="*50)
    print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Current Time: {weather_data['timestamp']}")
    print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

def save_weather_data_to_file(weather_data, filename="weather_data.json"):
    """
    Save weather data to a JSON file.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"\nWeather data saved to {filename}")
    except IOError as e:
        print(f"Error saving to file: {str(e)}")

if __name__ == "__main__":
    try:
        city = input("Enter city name: ").strip()
        if not city:
            city = "London"
            print(f"No city entered, using default: {city}")
        
        weather_info = get_weather_data(city)
        display_weather_data(weather_info)
        
        save_option = input("\nSave this data to file? (y/n): ").strip().lower()
        if save_option == 'y':
            filename = input("Enter filename (default: weather_data.json): ").strip()
            if not filename:
                filename = "weather_data.json"
            save_weather_data_to_file(weather_info, filename)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your API key and internet connection.")
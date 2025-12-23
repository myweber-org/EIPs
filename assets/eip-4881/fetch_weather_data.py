import requests
import json
from datetime import datetime

def get_weather(api_key, city):
    """
    Fetch current weather data for a given city.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['cod'] != 200:
            return {'error': data.get('message', 'Unknown error')}
        
        weather_info = {
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
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return {'error': f'Network error: {str(e)}'}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return {'error': f'Data parsing error: {str(e)}'}

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if 'error' in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    print("\n" + "="*50)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*50)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    display_weather(weather)
import requests
import json
import os

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a specified city.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str): OpenWeatherMap API key. If None, tries to get from env var.
    
    Returns:
        dict: Weather data if successful, None otherwise.
    """
    if api_key is None:
        api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Error: API key not provided")
        return None
    
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
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'visibility': data.get('visibility', 'N/A')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    
    Args:
        weather_data (dict): Weather data dictionary
    """
    if not weather_data:
        print("No weather data available")
        return
    
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Conditions: {weather_data['weather']} ({weather_data['description']})")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")
    if weather_data['visibility'] != 'N/A':
        print(f"  Visibility: {weather_data['visibility']} meters")

if __name__ == "__main__":
    # Example usage
    city = "London"
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    weather = get_weather(city, api_key)
    display_weather(weather)
import requests
import os

def get_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city.
    
    Args:
        city_name (str): Name of the city to get weather for
        api_key (str, optional): OpenWeatherMap API key. If not provided,
                               will try to get from WEATHER_API_KEY environment variable.
    
    Returns:
        dict: Weather data if successful, None otherwise
    """
    if api_key is None:
        api_key = os.environ.get('WEATHER_API_KEY')
    
    if not api_key:
        raise ValueError("API key must be provided or set in WEATHER_API_KEY environment variable")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    
    Args:
        weather_data (dict): Weather data from OpenWeatherMap API
    """
    if not weather_data:
        print("No weather data available")
        return
    
    try:
        city = weather_data['name']
        country = weather_data['sys']['country']
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        description = weather_data['weather'][0]['description']
        wind_speed = weather_data['wind']['speed']
        
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Conditions: {description}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
        
    except KeyError as e:
        print(f"Unexpected data format: missing key {e}")

if __name__ == "__main__":
    # Example usage
    city = "London"
    weather = get_weather(city)
    
    if weather:
        display_weather(weather)
    else:
        print(f"Could not retrieve weather for {city}")
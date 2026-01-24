import requests
import os
from typing import Optional, Dict, Any

def get_current_weather(city_name: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        city_name: Name of the city to get weather for.
        api_key: OpenWeatherMap API key. If None, tries to get from env var.
    
    Returns:
        Dictionary containing weather data or None if request fails.
    """
    if api_key is None:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided or set in OPENWEATHER_API_KEY environment variable")
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": city_name,
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data: Dict[str, Any]) -> None:
    """
    Display formatted weather information.
    
    Args:
        weather_data: Weather data dictionary from OpenWeatherMap API.
    """
    if weather_data is None:
        print("No weather data available.")
        return
    
    city = weather_data.get("name", "Unknown")
    country = weather_data.get("sys", {}).get("country", "")
    temp = weather_data.get("main", {}).get("temp", 0)
    feels_like = weather_data.get("main", {}).get("feels_like", 0)
    humidity = weather_data.get("main", {}).get("humidity", 0)
    description = weather_data.get("weather", [{}])[0].get("description", "Unknown")
    
    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Humidity: {humidity}%")
    print(f"  Conditions: {description.capitalize()}")

if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        weather = get_current_weather("London", api_key)
        display_weather(weather)
    else:
        print("Please set OPENWEATHER_API_KEY environment variable.")
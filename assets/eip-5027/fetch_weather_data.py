import requests
import os

def get_current_weather(city_name, api_key=None):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if api_key is None:
            raise ValueError("API key must be provided or set as OPENWEATHER_API_KEY environment variable")

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        return weather_info

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    api_key = "your_api_key_here"  # Replace with your actual API key
    city = "London"
    weather = get_current_weather(city, api_key)
    if weather:
        print(f"Weather in {weather['city']}, {weather['country']}:")
        print(f"Temperature: {weather['temperature']}°C")
        print(f"Feels like: {weather['feels_like']}°C")
        print(f"Humidity: {weather['humidity']}%")
        print(f"Weather: {weather['weather']}")
        print(f"Wind Speed: {weather['wind_speed']} m/s")
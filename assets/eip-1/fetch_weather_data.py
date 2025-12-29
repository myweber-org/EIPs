import requests
import json

def get_weather_data(api_key, city):
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
            return None
            
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data:
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Conditions: {weather_data['weather']} - {weather_data['description']}")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    else:
        print("Unable to retrieve weather data.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather_data(API_KEY, CITY)
    display_weather(weather)
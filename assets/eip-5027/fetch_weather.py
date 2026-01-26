
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch weather data from OpenWeatherMap API"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WeatherFetcher/1.0',
            'Accept': 'application/json'
        })
    
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Fetch current weather for a given city"""
        query = city
        if country_code:
            query = f"{city},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            return {
                'error': True,
                'message': f"Network error: {str(e)}"
            }
        except json.JSONDecodeError:
            return {
                'error': True,
                'message': "Invalid response from API"
            }
    
    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw API response into structured format"""
        if raw_data.get('cod') != 200:
            return {
                'error': True,
                'message': raw_data.get('message', 'Unknown API error')
            }
        
        main = raw_data.get('main', {})
        weather = raw_data.get('weather', [{}])[0]
        wind = raw_data.get('wind', {})
        
        return {
            'error': False,
            'location': {
                'city': raw_data.get('name'),
                'country': raw_data.get('sys', {}).get('country'),
                'coordinates': raw_data.get('coord', {})
            },
            'weather': {
                'temperature': main.get('temp'),
                'feels_like': main.get('feels_like'),
                'humidity': main.get('humidity'),
                'pressure': main.get('pressure'),
                'description': weather.get('description'),
                'main': weather.get('main'),
                'wind_speed': wind.get('speed'),
                'wind_direction': wind.get('deg')
            },
            'timestamp': datetime.utcnow().isoformat(),
            'raw_timestamp': raw_data.get('dt'),
            'timezone': raw_data.get('timezone')
        }
    
    def format_weather_report(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data into human-readable report"""
        if weather_data.get('error'):
            return f"Error: {weather_data.get('message')}"
        
        loc = weather_data['location']
        w = weather_data['weather']
        
        report_lines = [
            f"Weather Report for {loc['city']}, {loc['country']}",
            "=" * 40,
            f"Temperature: {w['temperature']}°C (feels like {w['feels_like']}°C)",
            f"Conditions: {w['description'].title()}",
            f"Humidity: {w['humidity']}%",
            f"Pressure: {w['pressure']} hPa",
            f"Wind: {w['wind_speed']} m/s at {w['wind_direction']}°",
            f"Report generated: {weather_data['timestamp']}"
        ]
        
        return "\n".join(report_lines)

def main():
    """Example usage of the WeatherFetcher class"""
    api_key = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(api_key)
    
    # Fetch weather for London
    weather_data = fetcher.get_weather("London", "UK")
    
    if not weather_data.get('error'):
        print(fetcher.format_weather_report(weather_data))
        
        # Save to JSON file
        with open('weather_data.json', 'w') as f:
            json.dump(weather_data, f, indent=2)
        print("\nData saved to weather_data.json")
    else:
        print(f"Failed to fetch weather: {weather_data.get('message')}")

if __name__ == "__main__":
    main()
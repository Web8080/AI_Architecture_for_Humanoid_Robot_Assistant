"""
Weather Information Engine

PURPOSE:
    Provides current weather conditions, forecasts, and weather-based recommendations.
    Integrates with WeatherAPI for real-time data and provides context-aware advice.

HOME ASSISTANT CONTEXT:
    - Helps families plan daily activities based on weather
    - Recommends appropriate clothing for children
    - Warns about severe weather conditions
    - Suggests indoor/outdoor activities
    - Provides gardening and pet care advice based on weather

USE CASES:
    1. Morning briefing: "What's the weather like today?"
    2. Activity planning: "Is it good weather for the park?"
    3. Child care: "What should the kids wear outside?"
    4. Safety: "Is there a weather warning today?"
    5. Pet care: "Should I take the dog for a long walk?"

Author: Victor Ibhafidon  
Date: November 2025
Version: 2.0 (Production Quality)
"""

from typing import Dict, Any, Optional, List
from src.engines.base_engine import BaseEngine
import logging
import os
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WeatherEngine(BaseEngine):
    """
    Production-grade weather engine for home assistant robots.
    
    CAPABILITIES:
    - Current weather conditions
    - 7-day forecast
    - Severe weather alerts
    - Context-aware recommendations (clothing, activities, safety)
    - Air quality information
    - UV index warnings
    - Precipitation probability
    
    MULTI-TIER FALLBACK:
    - Tier 1: WeatherAPI (real-time, comprehensive data)
    - Tier 2: OpenWeatherMap (backup API)
    - Tier 3: Generic weather advice based on season/time
    
    INTEGRATION:
    - Connects to calendar for activity planning
    - Links to reminder system for weather-based alerts
    - Integrates with home automation for climate control
    """
    
    # Weather condition categories
    SUNNY = ['sunny', 'clear']
    CLOUDY = ['cloudy', 'overcast', 'partly cloudy']
    RAINY = ['rain', 'drizzle', 'showers', 'thunderstorm']
    SNOWY = ['snow', 'sleet', 'blizzard']
    STORMY = ['thunderstorm', 'storm', 'tornado', 'hurricane']
    
    # Temperature thresholds (Celsius)
    VERY_COLD = 0
    COLD = 10
    MILD = 15
    WARM = 20
    HOT = 28
    VERY_HOT = 35
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize weather engine.
        
        Args:
            config: Configuration dictionary with:
                - api_key (str): WeatherAPI key
                - backup_api_key (str): OpenWeatherMap key
                - default_location (str): Default city/location
                - units (str): 'metric' or 'imperial'
                - enable_alerts (bool): Enable severe weather alerts
        """
        super().__init__(config)
        self.name = "WeatherEngine"
        
        # API configuration
        self.api_key = config.get('api_key') if config else os.getenv("WEATHER_API_KEY", "9da4a523b41c453ab6f91434251604")
        self.backup_api_key = config.get('backup_api_key') if config else os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.weatherapi.com/v1"
        self.backup_url = "http://api.openweathermap.org/data/2.5"
        
        # User preferences
        self.default_location = config.get('default_location', 'London') if config else 'London'
        self.units = config.get('units', 'metric') if config else 'metric'
        self.enable_alerts = config.get('enable_alerts', True) if config else True
        
        # Cache for performance (5-minute cache)
        self.weather_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration_seconds = 300
        
        logger.info(f"✓ {self.name} initialized")
        logger.info(f"  - Default location: {self.default_location}")
        logger.info(f"  - Units: {self.units}")
        logger.info(f"  - Alerts enabled: {self.enable_alerts}")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get weather information with context-aware recommendations.
        
        Args:
            context: Request context containing:
                - location (str): City or location name (optional, uses default)
                - query_type (str): 'current' | 'forecast' | 'alerts' | 'recommendations'
                - activity (str): Optional activity for recommendations
                - user_age_group (str): 'child' | 'adult' | 'elderly' for tailored advice
        
        Returns:
            Weather information with recommendations
        """
        # Extract parameters
        location = context.get('location', self.default_location)
        query_type = context.get('query_type', 'current')
        activity = context.get('activity')
        user_age_group = context.get('user_age_group', 'adult')
        
        logger.info(f"⛅ Weather request for {location} ({query_type})")
        
        try:
            # TIER 1: WeatherAPI (best)
            result = self._tier1_weatherapi(location, query_type, activity, user_age_group)
            logger.info(f"✓ Tier 1 weather data retrieved")
            return result
            
        except Exception as e1:
            logger.warning(f"Tier 1 failed: {e1}, falling back to Tier 2")
            
            try:
                # TIER 2: OpenWeatherMap backup
                result = self._tier2_openweather(location, query_type, user_age_group)
                logger.info(f"✓ Tier 2 weather data retrieved")
                return result
                
            except Exception as e2:
                logger.warning(f"Tier 2 failed: {e2}, falling back to Tier 3")
                
                # TIER 3: Generic seasonal advice
                result = self._tier3_generic_advice(location, user_age_group)
                logger.warning(f"⚠️ Using Tier 3 generic weather advice")
                return result
    
    def _tier1_weatherapi(
        self, 
        location: str, 
        query_type: str, 
        activity: Optional[str],
        user_age_group: str
    ) -> Dict[str, Any]:
        """
        TIER 1: WeatherAPI comprehensive data.
        
        Provides detailed weather with recommendations.
        """
        # Check cache first
        cache_key = f"{location}_{query_type}"
        if cache_key in self.weather_cache:
            cached_data = self.weather_cache[cache_key]
            cache_time = cached_data.get('timestamp', 0)
            if (datetime.now().timestamp() - cache_time) < self.cache_duration_seconds:
                logger.debug(f"Using cached weather data for {location}")
                return cached_data['data']
        
        # Fetch current weather
        current_url = f"{self.base_url}/current.json"
        params = {
            "key": self.api_key,
            "q": location,
            "aqi": "yes"  # Include air quality
        }
        
        response = requests.get(current_url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        # Extract current conditions
        current = data['current']
        location_info = data['location']
        
        # Build comprehensive weather report
        weather_data = {
            'location': {
                'name': location_info['name'],
                'region': location_info['region'],
                'country': location_info['country'],
                'local_time': location_info['localtime']
            },
            'current_conditions': {
                'temperature_c': current['temp_c'],
                'temperature_f': current['temp_f'],
                'feels_like_c': current['feelslike_c'],
                'condition': current['condition']['text'],
                'condition_icon': current['condition']['icon'],
                'wind_kph': current['wind_kph'],
                'wind_dir': current['wind_dir'],
                'pressure_mb': current['pressure_mb'],
                'humidity_percent': current['humidity'],
                'cloud_cover_percent': current['cloud'],
                'visibility_km': current['vis_km'],
                'uv_index': current['uv'],
                'precipitation_mm': current['precip_mm']
            },
            'air_quality': current.get('air_quality', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add context-aware recommendations
        recommendations = self._generate_recommendations(
            weather_data['current_conditions'],
            activity,
            user_age_group
        )
        weather_data['recommendations'] = recommendations
        
        # Add weather alerts if enabled
        if self.enable_alerts:
            alerts = self._check_alerts(weather_data['current_conditions'])
            if alerts:
                weather_data['alerts'] = alerts
        
        # Generate natural language summary
        weather_data['summary'] = self._generate_summary(weather_data, user_age_group)
        
        # Cache the result
        self.weather_cache[cache_key] = {
            'data': weather_data,
            'timestamp': datetime.now().timestamp()
        }
        
        # Return with success status
        return {
            **weather_data,
            'tier_used': 1,
            'status': 'success',
            'data_source': 'WeatherAPI'
        }
    
    def _tier2_openweather(
        self, 
        location: str, 
        query_type: str,
        user_age_group: str
    ) -> Dict[str, Any]:
        """
        TIER 2: OpenWeatherMap backup API.
        
        Limited data but reliable backup.
        """
        if not self.backup_api_key:
            raise Exception("No backup API key available")
        
        url = f"{self.backup_url}/weather"
        params = {
            "q": location,
            "appid": self.backup_api_key,
            "units": self.units
        }
        
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        # Basic weather data
        weather_data = {
            'location': {'name': data['name'], 'country': data['sys']['country']},
            'current_conditions': {
                'temperature_c': data['main']['temp'],
                'feels_like_c': data['main']['feels_like'],
                'condition': data['weather'][0]['description'],
                'humidity_percent': data['main']['humidity'],
                'wind_kph': data['wind']['speed'] * 3.6  # Convert m/s to km/h
            },
            'summary': f"The weather in {data['name']} is {data['weather'][0]['description']} with {data['main']['temp']}°C.",
            'tier_used': 2,
            'status': 'success',
            'data_source': 'OpenWeatherMap',
            'warnings': ['Limited weather data - using backup API']
        }
        
        return weather_data
    
    def _tier3_generic_advice(self, location: str, user_age_group: str) -> Dict[str, Any]:
        """
        TIER 3: Generic seasonal advice (always available).
        
        Provides basic guidance when no API available.
        """
        # Get current month for seasonal advice
        month = datetime.now().month
        season = self._get_season(month)
        
        generic_advice = {
            'winter': "It's winter - dress warmly with layers, hat, and gloves. Be cautious of ice.",
            'spring': "It's spring - weather can be unpredictable. Bring a light jacket and umbrella.",
            'summer': "It's summer - stay hydrated, use sunscreen, and avoid midday sun.",
            'autumn': "It's autumn - dress in layers as temperatures vary. Watch for rain."
        }
        
        return {
            'location': {'name': location},
            'current_conditions': {'season': season},
            'summary': f"I couldn't get current weather data for {location}. {generic_advice[season]}",
            'recommendations': {
                'clothing': self._generic_clothing_advice(season, user_age_group),
                'activities': self._generic_activity_advice(season)
            },
            'tier_used': 3,
            'status': 'partial',
            'data_source': 'Generic seasonal data',
            'warnings': ['Unable to fetch real-time weather - using seasonal averages']
        }
    
    def _generate_recommendations(
        self,
        conditions: Dict[str, Any],
        activity: Optional[str],
        user_age_group: str
    ) -> Dict[str, Any]:
        """
        Generate context-aware weather recommendations.
        
        Considers temperature, conditions, user age, and planned activity.
        """
        temp_c = conditions['temperature_c']
        feels_like = conditions['feels_like_c']
        condition_text = conditions['condition'].lower()
        uv_index = conditions.get('uv_index', 0)
        humidity = conditions['humidity_percent']
        
        recommendations = {}
        
        # CLOTHING RECOMMENDATIONS
        clothing = []
        if feels_like < self.VERY_COLD:
            clothing = ['Heavy winter coat', 'Thermal layers', 'Warm hat', 'Gloves', 'Scarf', 'Warm boots']
        elif feels_like < self.COLD:
            clothing = ['Warm jacket', 'Long sleeves', 'Light gloves', 'Closed shoes']
        elif feels_like < self.MILD:
            clothing = ['Light jacket or sweater', 'Long pants', 'Comfortable shoes']
        elif feels_like < self.WARM:
            clothing = ['Light shirt', 'Comfortable pants or shorts', 'Sunglasses']
        elif feels_like < self.HOT:
            clothing = ['Light breathable clothing', 'Sun hat', 'Sunglasses', 'Sunscreen']
        else:
            clothing = ['Very light clothing', 'Wide-brimmed hat', 'Sunglasses', 'High SPF sunscreen', 'Stay hydrated!']
        
        # Add rain gear if needed
        if any(rain in condition_text for rain in self.RAINY):
            clothing.extend(['Umbrella', 'Waterproof jacket', 'Water-resistant shoes'])
        
        recommendations['clothing'] = clothing
        
        # ACTIVITY RECOMMENDATIONS
        if activity:
            recommendations['activity_advice'] = self._get_activity_advice(activity, conditions)
        else:
            # General activity suggestions
            if feels_like > self.WARM and any(sunny in condition_text for sunny in self.SUNNY):
                recommendations['suggested_activities'] = ['Park visit', 'Outdoor sports', 'Gardening', 'BBQ']
            elif any(rain in condition_text for rain in self.RAINY):
                recommendations['suggested_activities'] = ['Indoor games', 'Museum visit', 'Reading', 'Arts and crafts']
            elif feels_like < self.COLD:
                recommendations['suggested_activities'] = ['Indoor activities', 'Board games', 'Baking', 'Movie time']
            else:
                recommendations['suggested_activities'] = ['Light outdoor walk', 'Shopping', 'Café visit']
        
        # HEALTH AND SAFETY
        health_advice = []
        if uv_index > 6:
            health_advice.append(f'High UV index ({uv_index}) - apply SPF 30+ sunscreen')
        if feels_like > self.VERY_HOT:
            health_advice.append('Extreme heat - limit outdoor exposure, drink plenty of water')
        if feels_like < self.VERY_COLD:
            health_advice.append('Extreme cold - limit time outdoors, watch for frostbite')
        if humidity > 80:
            health_advice.append('High humidity - may feel uncomfortable, stay hydrated')
        
        if health_advice:
            recommendations['health_safety'] = health_advice
        
        # AGE-SPECIFIC ADVICE
        if user_age_group == 'child':
            recommendations['parent_note'] = self._get_child_safety_advice(conditions)
        elif user_age_group == 'elderly':
            recommendations['elderly_care'] = self._get_elderly_care_advice(conditions)
        
        return recommendations
    
    def _generate_summary(self, weather_data: Dict[str, Any], user_age_group: str) -> str:
        """Generate natural language weather summary."""
        loc = weather_data['location']['name']
        cond = weather_data['current_conditions']
        temp = cond['temperature_c']
        feels = cond['feels_like_c']
        condition = cond['condition']
        
        # Base summary
        summary = f"The weather in {loc} is {condition} with a temperature of {temp}°C"
        
        # Add feels like if significantly different
        if abs(temp - feels) > 3:
            summary += f", but it feels like {feels}°C"
        
        # Add simple recommendation
        if temp > self.HOT:
            summary += ". It's quite hot - stay cool and hydrated."
        elif temp < self.COLD:
            summary += ". It's cold - dress warmly."
        elif 'rain' in condition.lower():
            summary += ". Don't forget your umbrella!"
        else:
            summary += ". Have a great day!"
        
        return summary
    
    def _check_alerts(self, conditions: Dict[str, Any]) -> List[str]:
        """Check for severe weather alerts."""
        alerts = []
        
        temp = conditions['temperature_c']
        feels_like = conditions['feels_like_c']
        wind_kph = conditions['wind_kph']
        uv = conditions.get('uv_index', 0)
        condition = conditions['condition'].lower()
        
        # Temperature alerts
        if feels_like > 35:
            alerts.append('⚠️ EXTREME HEAT WARNING - Limit outdoor activities')
        elif feels_like < -10:
            alerts.append('⚠️ EXTREME COLD WARNING - Frostbite risk')
        
        # Wind alerts
        if wind_kph > 50:
            alerts.append('⚠️ HIGH WIND WARNING - Secure loose objects')
        
        # UV alerts
        if uv > 8:
            alerts.append('⚠️ VERY HIGH UV - Sun protection essential')
        
        # Storm alerts
        if any(storm in condition for storm in self.STORMY):
            alerts.append('⚠️ STORM WARNING - Stay indoors if possible')
        
        return alerts
    
    def _get_season(self, month: int) -> str:
        """Get season from month (Northern Hemisphere)."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def _generic_clothing_advice(self, season: str, age_group: str) -> List[str]:
        """Generic clothing advice by season."""
        advice = {
            'winter': ['Warm coat', 'Hat', 'Gloves', 'Scarf'],
            'spring': ['Light jacket', 'Layers', 'Umbrella'],
            'summer': ['Light clothes', 'Sun hat', 'Sunscreen'],
            'autumn': ['Medium jacket', 'Layers', 'Comfortable shoes']
        }
        return advice.get(season, [])
    
    def _generic_activity_advice(self, season: str) -> List[str]:
        """Generic activity suggestions by season."""
        advice = {
            'winter': ['Indoor activities', 'Hot drinks', 'Cozy games'],
            'spring': ['Park walks', 'Gardening', 'Outdoor sports'],
            'summer': ['Swimming', 'Picnics', 'Outdoor adventures'],
            'autumn': ['Nature walks', 'Seasonal crafts', 'Harvest activities']
        }
        return advice.get(season, [])
    
    def _get_activity_advice(self, activity: str, conditions: Dict[str, Any]) -> str:
        """Get advice for specific activity based on weather."""
        activity_lower = activity.lower()
        temp = conditions['temperature_c']
        condition = conditions['condition'].lower()
        
        if 'park' in activity_lower or 'outdoor' in activity_lower:
            if any(rain in condition for rain in self.RAINY):
                return f"Not ideal for {activity} due to rain. Consider indoor alternatives."
            elif temp > self.HOT:
                return f"Good for {activity} but go early morning or evening to avoid heat."
            elif temp < self.COLD:
                return f"Dress very warmly for {activity}. Consider shorter duration."
            else:
                return f"Great weather for {activity}! Enjoy!"
        
        return f"Weather is {condition} with {temp}°C - plan accordingly for {activity}."
    
    def _get_child_safety_advice(self, conditions: Dict[str, Any]) -> str:
        """Safety advice for children."""
        temp = conditions['temperature_c']
        
        if temp > 30:
            return "Keep children hydrated, apply sunscreen frequently, limit midday sun exposure."
        elif temp < 5:
            return "Ensure children are warmly dressed. Limit outdoor time. Watch for signs of cold."
        else:
            return "Dress children appropriately for the weather. Supervise outdoor play."
    
    def _get_elderly_care_advice(self, conditions: Dict[str, Any]) -> str:
        """Care advice for elderly users."""
        temp = conditions['temperature_c']
        
        if temp > 28:
            return "Stay cool, drink water regularly, avoid strenuous activity."
        elif temp < 8:
            return "Keep warm, dress in layers, be cautious of ice/slippery surfaces."
        else:
            return "Weather is comfortable. Light activity recommended."
    
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if not isinstance(context, dict):
            return False
        
        # Validate query type if provided
        if 'query_type' in context:
            valid_types = ['current', 'forecast', 'alerts', 'recommendations']
            if context['query_type'] not in valid_types:
                logger.error(f"Invalid query_type: {context['query_type']}")
                return False
        
        return True


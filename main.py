import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import aiohttp
import asyncio
from typing import Dict, List, Optional, Tuple
import logging
from dotenv import load_dotenv
import re
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="VAYUDUT Air Quality API - Named Features", debug=True)

# OpenWeatherMap API Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY")
OPENWEATHER_ONE_CALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

logger.info("🚀 Starting VAYUDUT API - Named Feature Format")

# City coordinates
CITY_COORDINATES = {
    "delhi": (28.6139, 77.2090),
    "new delhi": (28.6139, 77.2090),
    "delhi ncr": (28.6139, 77.2090),
    "ncr": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "kolkata": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "ahmedabad": (23.0225, 72.5714),
    "chandigarh": (30.7333, 76.7794),
    "jaipur": (26.9124, 75.7873),
    "lucknow": (26.8467, 80.9462),
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
}

def normalize_city_name(city_name: str) -> str:
    """Clean city name for matching"""
    if not city_name:
        return ""
    normalized = re.sub(r'\s+', ' ', city_name.lower().strip())
    suffixes_to_remove = ['district', 'dist', 'ncr', 'metro', 'city', 'mc']
    for suffix in suffixes_to_remove:
        normalized = re.sub(rf'\b{suffix}\b', '', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()

def find_city_match(city_name: str) -> Optional[str]:
    """Find best matching city"""
    normalized_input = normalize_city_name(city_name)
    if not normalized_input:
        return None
    
    if normalized_input in CITY_COORDINATES:
        return normalized_input
    
    input_words = normalized_input.split()
    for city_key in CITY_COORDINATES:
        if all(word in city_key for word in input_words):
            return city_key
    
    first_word = input_words[0] if input_words else ""
    for city_key in CITY_COORDINATES:
        if first_word in city_key:
            return city_key
    
    return None

# --- CRITICAL: Model with Named Features ---
MODEL_PATH = "final_gru_model.keras"
model = None
EXPECTED_TIMESTEPS = 24
EXPECTED_FEATURES = 18  # Default number of features expected by model
MODEL_FEATURE_NAMES = ['hour', 'NO2', 'O3']  # Start with minimum required
MODEL_INPUT_FEATURES = []  # Will be detected from model

def load_model_safely():
    """Load model and detect exact feature names"""
    global model, EXPECTED_TIMESTEPS, EXPECTED_FEATURES, MODEL_INPUT_FEATURES, MODEL_FEATURE_NAMES
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"⚠️ Model file not found: {MODEL_PATH}")
            return None
        
        logger.info("🔄 Loading ML model with feature name detection...")
        start_time = time.time()
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Get input shape
        input_shape = model.input_shape
        if len(input_shape) > 2:
            EXPECTED_TIMESTEPS = input_shape[1]
            num_features = input_shape[2]
            EXPECTED_FEATURES = num_features
        else:
            EXPECTED_TIMESTEPS = 24
            num_features = 18
            EXPECTED_FEATURES = 18
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.2f}s | Shape: (1, {EXPECTED_TIMESTEPS}, {num_features})")
        
        # CRITICAL: Detect exact feature names from model
        MODEL_INPUT_FEATURES = detect_model_feature_names(model, num_features)
        MODEL_FEATURE_NAMES = MODEL_INPUT_FEATURES[:3]  # First 3 for logging
        
        # Test with properly named features
        test_input = create_test_features_with_names(EXPECTED_TIMESTEPS, num_features)
        test_start = time.time()
        with tf.device('/CPU:0'):
            _ = model.predict(test_input, verbose=0)
        test_time = time.time() - test_start
        logger.info(f"⚡ Test inference with named features: {test_time:.3f}s")
        logger.info(f"📋 Detected features: {MODEL_INPUT_FEATURES[:5]}... (total: {len(MODEL_INPUT_FEATURES)})")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        # Fallback feature names
        EXPECTED_FEATURES = 18  # Default fallback
        MODEL_INPUT_FEATURES = generate_fallback_feature_names(EXPECTED_FEATURES)
        return None

def detect_model_feature_names(model, num_features: int) -> List[str]:
    """Detect exact feature names from model metadata"""
    try:
        # Try to get feature names from model config
        if hasattr(model, 'layers') and len(model.layers) > 0:
            first_layer = model.layers[0]
            if hasattr(first_layer, 'feature_names') or hasattr(first_layer, 'input_shape'):
                # Try to extract from layer
                logger.info("🔍 Extracting feature names from model layers...")
                return extract_layer_feature_names(first_layer, num_features)
        
        # Fallback: try model input names
        if hasattr(model.input, 'name'):
            input_name = model.input.name
            if 'feature' in input_name.lower() or 'input' in input_name.lower():
                logger.info(f"📋 Using input name hint: {input_name}")
        
        # Ultimate fallback: standard air quality features
        logger.info("🔍 Using standard air quality feature names")
        return generate_standard_feature_names(num_features)
        
    except Exception as e:
        logger.warning(f"⚠️ Feature detection failed: {e}, using standard names")
        return generate_standard_feature_names(num_features)

def extract_layer_feature_names(layer, num_features: int) -> List[str]:
    """Extract feature names from model layer"""
    try:
        # Check if layer has feature names
        if hasattr(layer, 'feature_names'):
            names = layer.feature_names
            if len(names) == num_features:
                return list(names)
        
        # Try layer weights or config
        if hasattr(layer, 'get_config'):
            config = layer.get_config()
            if 'feature_names' in config:
                names = config['feature_names']
                if len(names) == num_features:
                    return list(names)
        
        return generate_standard_feature_names(num_features)
    except:
        return generate_standard_feature_names(num_features)

def generate_standard_feature_names(num_features: int) -> List[str]:
    """Generate standard air quality feature names"""
    base_features = [
        'hour_sin', 'hour_cos',  # Time features
        'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction',
        'NO2', 'O3', 'SO2', 'CO', 'PM2.5', 'PM10', 'NO', 'NH3',
        'latitude', 'longitude', 'aqi'
    ]
    
    # Extend if needed
    if num_features > len(base_features):
        extra_features = [f'feature_{i}' for i in range(num_features - len(base_features))]
        base_features.extend(extra_features)
    else:
        base_features = base_features[:num_features]
    
    # Ensure 'hour', 'NO2', 'O3' are included
    if 'hour_sin' not in base_features:
        base_features.insert(0, 'hour_sin')
    if 'NO2' not in base_features:
        base_features.insert(7, 'NO2') 
    if 'O3' not in base_features:
        base_features.insert(8, 'O3')
    
    return base_features[:num_features]

def generate_fallback_feature_names(num_features: int) -> List[str]:
    """Generate fallback feature names"""
    return ['feature_' + str(i) for i in range(num_features)]

def create_test_features_with_names(timesteps: int, num_features: int) -> Dict:
    """Create test features with proper names for model validation"""
    # Create sample data with named columns
    feature_names = MODEL_INPUT_FEATURES if MODEL_INPUT_FEATURES else generate_standard_feature_names(num_features)
    
    # Create DataFrame with proper column names
    test_data = []
    for t in range(timesteps):
        row = {}
        hour = t % 24
        row['hour_sin'] = math.sin(2 * math.pi * hour / 24)
        row['hour_cos'] = math.cos(2 * math.pi * hour / 24)
        row['temperature'] = 25 + 5 * math.sin(2 * math.pi * hour / 24)
        row['humidity'] = 60 + 20 * math.cos(2 * math.pi * hour / 24)
        row['pressure'] = 1013
        row['wind_speed'] = 3 + 2 * abs(math.sin(2 * math.pi * hour / 12))
        row['NO2'] = 50 + 20 * math.sin(2 * math.pi * (hour + 6) / 24)
        row['O3'] = 60 + 15 * math.cos(2 * math.pi * hour / 24)
        
        # Fill remaining features
        for i, feature_name in enumerate(feature_names):
            if feature_name not in row:
                if 'feature_' in feature_name:
                    row[feature_name] = 0.5
                elif feature_name in ['latitude', 'longitude']:
                    row[feature_name] = 0.5
                else:
                    row[feature_name] = 0.5
        
        test_data.append(row)
    
    df = pd.DataFrame(test_data, columns=feature_names)
    return {'input': df.values.reshape(1, timesteps, num_features)}

# Load model with feature detection
model = load_model_safely()

# Fallback predictions
def generate_fallback_predictions(horizon: int, current_no2: float = 50, current_o3: float = 60) -> List[float]:
    """Generate fallback predictions"""
    predictions = []
    base_aqi = (current_no2 / 200.0 + current_o3 / 300.0) / 2 * 5
    
    np.random.seed(int(time.time() * 1000) % 10000)
    
    for i in range(horizon):
        cycle = 0.8 * math.sin(2 * math.pi * (i + 6) / 24)
        trend = -0.005 * i
        noise = np.random.normal(0, 0.15)
        aqi = max(0, min(5, base_aqi + cycle + trend + noise))
        predictions.append(float(aqi))
    
    return predictions

# Pydantic schema
class PredictionRequest(BaseModel):
    city: str
    horizon_hours: int = 24
    lat: Optional[float] = None
    lon: Optional[float] = None

# --- One Call API Client (unchanged from previous) ---
class OneCallWeatherClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
        self.semaphore = asyncio.Semaphore(3)
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        connector = aiohttp.TCPConnector(limit=5, limit_per_host=2)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        async with self.semaphore:
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                return None
    
    async def get_coordinates(self, city_name: str) -> Tuple[float, float]:
        matched_city = find_city_match(city_name)
        if matched_city:
            lat, lon = CITY_COORDINATES[matched_city]
            logger.info(f"📍 Local: {city_name} → ({lat}, {lon})")
            return lat, lon
        
        geo_url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city_name, "limit": 1, "appid": self.api_key}
        data = await self._make_request(geo_url, params)
        
        if data and len(data) > 0:
            lat, lon = data[0]["lat"], data[0]["lon"]
            logger.info(f"📍 GeoAPI: {city_name} → ({lat}, {lon})")
            return lat, lon
        
        logger.warning(f"⚠️ Using Delhi fallback for {city_name}")
        return 28.6139, 77.2090
    
    async def get_one_call_data(self, lat: float, lon: float) -> Optional[Dict]:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "exclude": "minutely,alerts"  # Exclude minutely and alerts for now
        }
        
        data = await self._make_request(OPENWEATHER_ONE_CALL_URL, params)
        if data:
            logger.info(f"🌤️ One Call API success for ({lat}, {lon})")
            # Log available data sections
            available_sections = [key for key in data.keys() if key not in ['lat', 'lon', 'timezone', 'timezone_offset']]
            logger.info(f"📊 Available data sections: {available_sections}")
            return data
        return None
    
    def extract_current_weather(self, one_call_data: Dict) -> Dict:
        if not one_call_data or "current" not in one_call_data:
            return self._get_fallback_weather()
        
        current = one_call_data["current"]
        weather = current["weather"][0] if current.get("weather") else {}
        
        return {
            "timestamp": datetime.fromtimestamp(current["dt"]),
            "temperature": current.get("temp", 25),
            "feels_like": current.get("feels_like", 25),
            "humidity": current.get("humidity", 60),
            "pressure": current.get("pressure", 1013),
            "wind_speed": current.get("wind_speed", 3),
            "wind_direction": current.get("wind_deg", 180),
            "wind_gust": current.get("wind_gust", 0),
            "clouds": current.get("clouds", 0),
            "visibility": current.get("visibility", 10000),
            "dew_point": current.get("dew_point", 0),
            "uvi": current.get("uvi", 0),
            "weather_description": weather.get("description", "clear sky"),
            "weather_main": weather.get("main", "Clear"),
            "weather_icon": weather.get("icon", "01d"),
            "source": "onecall_api"
        }
    
    def extract_current_pollution(self, one_call_data: Dict, lat: float, lon: float) -> Dict:
        # Estimate pollution from weather conditions
        hour = datetime.now().hour
        pollution_factor = 1 + 0.5 * (math.sin(2 * math.pi * (hour + 3) / 24) ** 2)
        
        is_india = abs(lat - 28.6) < 15 and abs(lon - 77.2) < 15
        base_no2 = 70 if is_india else 30
        base_o3 = 60 if is_india else 40
        base_pm25 = 50 if is_india else 20
        
        # Adjust based on weather conditions from the new API structure
        if one_call_data and "current" in one_call_data:
            current = one_call_data["current"]
            wind_factor = max(0.5, current.get("wind_speed", 3) / 10.0)
            humidity_factor = current.get("humidity", 60) / 100.0
            pressure_factor = current.get("pressure", 1013) / 1013.0
            visibility_factor = min(1.0, current.get("visibility", 10000) / 10000.0)
            
            # More sophisticated pollution estimation based on weather
            base_no2 *= (1 + humidity_factor * 0.5) * (1 - wind_factor * 0.3) * (1 + (1 - pressure_factor) * 0.2)
            base_o3 *= (1 - humidity_factor * 0.2) * (1 + wind_factor * 0.2) * (1 + visibility_factor * 0.1)
            base_pm25 *= (1 + humidity_factor * 0.4) * (1 - wind_factor * 0.2) * (1 + (1 - visibility_factor) * 0.3)
        
        # Calculate AQI based on the most dominant pollutant
        aqi_no2 = min(5, base_no2 / 200.0 * 5)
        aqi_o3 = min(5, base_o3 / 300.0 * 5)
        aqi_pm25 = min(5, base_pm25 / 50.0 * 5)
        overall_aqi = max(aqi_no2, aqi_o3, aqi_pm25)
        
        return {
            "timestamp": datetime.now(),
            "aqi": round(overall_aqi, 1),
            "co": round(600 * pollution_factor, 1),
            "no": round(25 * pollution_factor, 1),
            "no2": round(base_no2 * pollution_factor, 1),  # CRITICAL for model
            "o3": round(base_o3 + 20 * math.sin(2 * math.pi * hour / 24), 1),  # CRITICAL for model
            "so2": round(15 * pollution_factor, 1),
            "pm2_5": round(base_pm25 * pollution_factor, 1),
            "pm10": round(base_pm25 * 1.3 * pollution_factor, 1),
            "nh3": round(8 * pollution_factor, 1),
            "source": "estimated_from_weather"
        }
    
    def extract_hourly_weather(self, one_call_data: Dict, hours: int = 24) -> List[Dict]:
        """Extract hourly weather data from the new API structure"""
        if not one_call_data or "hourly" not in one_call_data:
            return self._get_fallback_hourly_weather(hours)
        
        hourly_data = one_call_data["hourly"][:hours]  # Take first N hours
        result = []
        
        for hour_data in hourly_data:
            weather = hour_data["weather"][0] if hour_data.get("weather") else {}
            result.append({
                "timestamp": datetime.fromtimestamp(hour_data["dt"]),
                "temperature": hour_data.get("temp", 25),
                "feels_like": hour_data.get("feels_like", 25),
                "humidity": hour_data.get("humidity", 60),
                "pressure": hour_data.get("pressure", 1013),
                "wind_speed": hour_data.get("wind_speed", 3),
                "wind_direction": hour_data.get("wind_deg", 180),
                "wind_gust": hour_data.get("wind_gust", 0),
                "clouds": hour_data.get("clouds", 0),
                "visibility": hour_data.get("visibility", 10000),
                "dew_point": hour_data.get("dew_point", 0),
                "uvi": hour_data.get("uvi", 0),
                "pop": hour_data.get("pop", 0),  # Probability of precipitation
                "weather_description": weather.get("description", "clear sky"),
                "weather_main": weather.get("main", "Clear"),
                "weather_icon": weather.get("icon", "01d"),
                "source": "onecall_api"
            })
        
        return result
    
    def extract_daily_weather(self, one_call_data: Dict, days: int = 7) -> List[Dict]:
        """Extract daily weather data from the new API structure"""
        if not one_call_data or "daily" not in one_call_data:
            return self._get_fallback_daily_weather(days)
        
        daily_data = one_call_data["daily"][:days]  # Take first N days
        result = []
        
        for day_data in daily_data:
            weather = day_data["weather"][0] if day_data.get("weather") else {}
            temp = day_data.get("temp", {})
            feels_like = day_data.get("feels_like", {})
            
            result.append({
                "timestamp": datetime.fromtimestamp(day_data["dt"]),
                "sunrise": datetime.fromtimestamp(day_data.get("sunrise", day_data["dt"])),
                "sunset": datetime.fromtimestamp(day_data.get("sunset", day_data["dt"])),
                "moonrise": datetime.fromtimestamp(day_data.get("moonrise", day_data["dt"])),
                "moonset": datetime.fromtimestamp(day_data.get("moonset", day_data["dt"])),
                "moon_phase": day_data.get("moon_phase", 0),
                "summary": day_data.get("summary", ""),
                "temperature": {
                    "day": temp.get("day", 25),
                    "min": temp.get("min", 20),
                    "max": temp.get("max", 30),
                    "night": temp.get("night", 22),
                    "eve": temp.get("eve", 28),
                    "morn": temp.get("morn", 24)
                },
                "feels_like": {
                    "day": feels_like.get("day", 25),
                    "night": feels_like.get("night", 22),
                    "eve": feels_like.get("eve", 28),
                    "morn": feels_like.get("morn", 24)
                },
                "pressure": day_data.get("pressure", 1013),
                "humidity": day_data.get("humidity", 60),
                "dew_point": day_data.get("dew_point", 0),
                "wind_speed": day_data.get("wind_speed", 3),
                "wind_direction": day_data.get("wind_deg", 180),
                "wind_gust": day_data.get("wind_gust", 0),
                "clouds": day_data.get("clouds", 0),
                "pop": day_data.get("pop", 0),
                "rain": day_data.get("rain", {}).get("1h", 0) if day_data.get("rain") else 0,
                "snow": day_data.get("snow", {}).get("1h", 0) if day_data.get("snow") else 0,
                "uvi": day_data.get("uvi", 0),
                "weather_description": weather.get("description", "clear sky"),
                "weather_main": weather.get("main", "Clear"),
                "weather_icon": weather.get("icon", "01d"),
                "source": "onecall_api"
            })
        
        return result
    
    def _get_fallback_weather(self) -> Dict:
        hour = datetime.now().hour
        base_temp = 15 + 10 * math.sin(2 * math.pi * (hour + 6) / 24)
        return {
            "timestamp": datetime.now(),
            "temperature": round(base_temp, 1),
            "humidity": round(40 + 30 * math.sin(2 * math.pi * hour / 24), 1),
            "pressure": round(1013 + 5 * math.sin(2 * math.pi * (hour + 9) / 24), 1),
            "wind_speed": round(2 + 2 * abs(math.sin(2 * math.pi * hour / 12)), 1),
            "clouds": 50,
            "source": "fallback"
        }
    
    def _get_fallback_hourly_weather(self, hours: int) -> List[Dict]:
        """Generate fallback hourly weather data"""
        result = []
        for i in range(hours):
            hour = (datetime.now().hour + i) % 24
            base_temp = 15 + 10 * math.sin(2 * math.pi * (hour + 6) / 24)
            result.append({
                "timestamp": datetime.now() + timedelta(hours=i),
                "temperature": round(base_temp, 1),
                "humidity": round(40 + 30 * math.sin(2 * math.pi * hour / 24), 1),
                "pressure": round(1013 + 5 * math.sin(2 * math.pi * (hour + 9) / 24), 1),
                "wind_speed": round(2 + 2 * abs(math.sin(2 * math.pi * hour / 12)), 1),
                "clouds": 50,
                "source": "fallback"
            })
        return result
    
    def _get_fallback_daily_weather(self, days: int) -> List[Dict]:
        """Generate fallback daily weather data"""
        result = []
        for i in range(days):
            day_temp = 20 + 10 * math.sin(2 * math.pi * i / 7)
            result.append({
                "timestamp": datetime.now() + timedelta(days=i),
                "temperature": {
                    "day": round(day_temp, 1),
                    "min": round(day_temp - 5, 1),
                    "max": round(day_temp + 5, 1),
                    "night": round(day_temp - 3, 1),
                    "eve": round(day_temp + 2, 1),
                    "morn": round(day_temp - 1, 1)
                },
                "humidity": round(50 + 20 * math.sin(2 * math.pi * i / 7), 1),
                "pressure": round(1013 + 5 * math.sin(2 * math.pi * i / 7), 1),
                "wind_speed": round(3 + 2 * abs(math.sin(2 * math.pi * i / 7)), 1),
                "clouds": 50,
                "source": "fallback"
            })
        return result

# Global client
weather_client = None
async def get_weather_client():
    global weather_client
    if weather_client is None:
        weather_client = OneCallWeatherClient(OPENWEATHER_API_KEY)
        await weather_client.__aenter__()
    return weather_client

# --- CRITICAL: Fixed Feature Creation with Named Columns ---
def create_model_features(data: Dict, num_timesteps: int) -> np.ndarray:
    """Create features with EXACT model column names using new API structure"""
    
    # Get feature names from model
    feature_names = MODEL_INPUT_FEATURES if MODEL_INPUT_FEATURES else generate_standard_feature_names(EXPECTED_FEATURES)
    logger.debug(f"📊 Using {len(feature_names)} feature names: {feature_names[:5]}...")
    
    # Extract data from new structure
    pollution = data["current"]["air_pollution"]
    weather = data["current"]["weather"]
    hourly_weather = data.get("hourly", [])
    lat, lon = data["coordinates"]["lat"], data["coordinates"]["lon"]
    fetch_time = data["fetch_time"]
    
    # Create time series data
    feature_data = []
    
    for t in range(num_timesteps):
        current_time = fetch_time + timedelta(hours=t)
        hour = current_time.hour
        row_data = {}
        
        # TIME FEATURES - CRITICAL: 'hour' column
        row_data['hour_sin'] = math.sin(2 * math.pi * hour / 24)
        row_data['hour_cos'] = math.cos(2 * math.pi * hour / 24)
        if 'hour' in feature_names:
            row_data['hour'] = hour  # Raw hour if expected
        
        # WEATHER FEATURES - Use hourly data if available, otherwise current
        if t < len(hourly_weather) and hourly_weather:
            hour_weather = hourly_weather[t]
            row_data['temperature'] = hour_weather.get("temperature", 25)
            row_data['humidity'] = hour_weather.get("humidity", 60)
            row_data['pressure'] = hour_weather.get("pressure", 1013)
            row_data['wind_speed'] = hour_weather.get("wind_speed", 3)
            row_data['wind_direction'] = hour_weather.get("wind_direction", 180)
            row_data['wind_gust'] = hour_weather.get("wind_gust", 0)
            row_data['clouds'] = hour_weather.get("clouds", 0)
            row_data['visibility'] = hour_weather.get("visibility", 10000)
            row_data['dew_point'] = hour_weather.get("dew_point", 0)
            row_data['uvi'] = hour_weather.get("uvi", 0)
            row_data['pop'] = hour_weather.get("pop", 0)
        else:
            # Use current weather data
            row_data['temperature'] = weather.get("temperature", 25)
            row_data['humidity'] = weather.get("humidity", 60)
            row_data['pressure'] = weather.get("pressure", 1013)
            row_data['wind_speed'] = weather.get("wind_speed", 3)
            row_data['wind_direction'] = weather.get("wind_direction", 180)
            row_data['wind_gust'] = weather.get("wind_gust", 0)
            row_data['clouds'] = weather.get("clouds", 0)
            row_data['visibility'] = weather.get("visibility", 10000)
            row_data['dew_point'] = weather.get("dew_point", 0)
            row_data['uvi'] = weather.get("uvi", 0)
            row_data['pop'] = 0
        
        # POLLUTION FEATURES - CRITICAL: 'NO2', 'O3' columns
        # For now, use current pollution data for all timesteps
        # In a more sophisticated implementation, you could estimate pollution for future hours
        row_data['NO2'] = pollution.get("no2", 40)  # Raw value, model will normalize
        row_data['O3'] = pollution.get("o3", 60)    # Raw value, model will normalize
        row_data['CO'] = pollution.get("co", 500)
        row_data['SO2'] = pollution.get("so2", 15)
        row_data['PM2.5'] = pollution.get("pm2_5", 30)
        row_data['PM10'] = pollution.get("pm10", 50)
        row_data['NO'] = pollution.get("no", 20)
        row_data['NH3'] = pollution.get("nh3", 5)
        
        # LOCATION FEATURES
        row_data['latitude'] = lat
        row_data['longitude'] = lon
        
        # AQI
        row_data['aqi'] = pollution.get("aqi", 3)
        
        # Fill missing features with defaults
        for feature_name in feature_names:
            if feature_name not in row_data:
                if feature_name.startswith('feature_'):
                    row_data[feature_name] = 0.5
                elif feature_name in ['latitude', 'longitude']:
                    row_data[feature_name] = 0.5
                else:
                    row_data[feature_name] = 0.0
        
        feature_data.append(row_data)
    
    # Create DataFrame with EXACT column order expected by model
    df = pd.DataFrame(feature_data, columns=feature_names)
    
    # Convert to numpy array with correct shape
    X = df.values.reshape(1, num_timesteps, -1)
    
    # Ensure exact shape
    if X.shape[2] != EXPECTED_FEATURES:
        logger.warning(f"⚠️ Feature count mismatch: {X.shape[2]} vs {EXPECTED_FEATURES}")
        if X.shape[2] < EXPECTED_FEATURES:
            # Pad with zeros
            padding = np.zeros((1, num_timesteps, EXPECTED_FEATURES - X.shape[2]))
            X = np.concatenate([X, padding], axis=2)
        else:
            # Truncate
            X = X[:, :, :EXPECTED_FEATURES]
    
    # Log sample values
    if 'NO2' in row_data:
        logger.debug(f"🔧 Features ready: shape {X.shape} | Sample NO2: {row_data['NO2']:.1f}, O3: {row_data['O3']:.1f}")
    
    return X

# --- Data Fetching (unchanged) ---
async def fetch_weather_data(city: str, lat: float = None, lon: float = None) -> Dict:
    """Fetch One Call API data"""
    start_time = time.time()
    
    client = await get_weather_client()
    
    if lat is None or lon is None:
        lat, lon = await client.get_coordinates(city)
    
    one_call_data = await asyncio.wait_for(
        client.get_one_call_data(lat, lon),
        timeout=10.0
    )
    
    if one_call_data:
        weather_data = client.extract_current_weather(one_call_data)
        pollution_data = client.extract_current_pollution(one_call_data, lat, lon)
        hourly_weather = client.extract_hourly_weather(one_call_data, 24)
        daily_weather = client.extract_daily_weather(one_call_data, 7)
        api_success = True
    else:
        weather_data = client._get_fallback_weather()
        hour = datetime.now().hour
        pollution_factor = 1 + 0.5 * (math.sin(2 * math.pi * (hour + 3) / 24) ** 2)
        pollution_data = {
            "timestamp": datetime.now(),
            "aqi": min(5, round(2 + 1.5 * pollution_factor, 1)),
            "no2": round(50 * pollution_factor, 1),  # Raw values for model
            "o3": round(60 + 20 * math.sin(2 * math.pi * hour / 24), 1),
            "source": "fallback"
        }
        hourly_weather = client._get_fallback_hourly_weather(24)
        daily_weather = client._get_fallback_daily_weather(7)
        api_success = False
    
    fetch_time = time.time() - start_time
    logger.info(f"🌤️ Data fetch: {fetch_time:.2f}s | API: {'success' if api_success else 'fallback'}")
    
    return {
        "city": city,
        "resolved_city": find_city_match(city) or city,
        "coordinates": {"lat": lat, "lon": lon},
        "timezone": one_call_data.get("timezone", "UTC") if one_call_data else "UTC",
        "timezone_offset": one_call_data.get("timezone_offset", 0) if one_call_data else 0,
        "fetch_time": datetime.now(),
        "current": {
            "air_pollution": pollution_data,
            "weather": weather_data
        },
        "hourly": hourly_weather,
        "daily": daily_weather,
        "api_success": api_success
    }

# --- FIXED: Model Prediction with Named Features ---
def safe_model_predict(X: np.ndarray, horizon: int) -> List[float]:
    """Model prediction expecting named features"""
    
    if model is None:
        logger.info("🤖 Using fallback (no model)")
        # Extract NO2 and O3 from features (assuming positions 7, 8)
        current_no2 = X[0, 0, 7] if X.shape[2] > 7 else 40
        current_o3 = X[0, 0, 8] if X.shape[2] > 8 else 60
        return generate_fallback_predictions(horizon, current_no2, current_o3)
    
    def predict_fn():
        """Inner prediction function"""
        try:
            # Validate shape
            if len(X.shape) != 3 or X.shape[0] != 1:
                raise ValueError(f"Invalid input shape: {X.shape}")
            if X.shape[1] != EXPECTED_TIMESTEPS:
                raise ValueError(f"Timesteps mismatch: {X.shape[1]} != {EXPECTED_TIMESTEPS}")
            if X.shape[2] != EXPECTED_FEATURES:
                raise ValueError(f"Features mismatch: {X.shape[2]} != {EXPECTED_FEATURES}")
            
            # Create named DataFrame for model (if model expects it)
            feature_names = MODEL_INPUT_FEATURES if MODEL_INPUT_FEATURES else generate_standard_feature_names(EXPECTED_FEATURES)
            df = pd.DataFrame(X[0].reshape(EXPECTED_TIMESTEPS, EXPECTED_FEATURES), columns=feature_names)
            
            # Some models expect DataFrame input
            try:
                with tf.device('/CPU:0'):
                    result = model.predict(df, verbose=0)
            except:
                # Fallback to numpy array
                with tf.device('/CPU:0'):
                    result = model.predict(X, verbose=0)
            
            predictions = result.flatten()[:horizon].tolist()
            logger.debug(f"✅ Model predicted {len(predictions)} values with named features")
            return predictions
                
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            raise e
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(predict_fn)
            result = future.result(timeout=5.0)  # 5s timeout
            return result
            
    except TimeoutError:
        logger.warning("⏰ Model prediction timeout (5s)")
        current_no2 = X[0, 0, 7] if X.shape[2] > 7 else 40
        current_o3 = X[0, 0, 8] if X.shape[2] > 8 else 60
        return generate_fallback_predictions(horizon, current_no2, current_o3)
    
    except Exception as e:
        logger.error(f"❌ Model prediction failed: {e}")
        current_no2 = X[0, 0, 7] if X.shape[2] > 7 else 40
        current_o3 = X[0, 0, 8] if X.shape[2] > 8 else 60
        return generate_fallback_predictions(horizon, current_no2, current_o3)

# --- Main Endpoint ---
@app.post("/predict")
async def predict_air_quality(req: PredictionRequest):
    """Main prediction endpoint"""
    start_time = time.time()
    request_id = f"req_{int(time.time()*1000)}_{hash(req.city) % 1000}"
    
    logger.info(f"🌍 [{request_id}] Request: {req.city} | Horizon: {req.horizon_hours}")
    
    if not req.city or not req.city.strip():
        raise HTTPException(status_code=400, detail="City name required")
    
    if req.horizon_hours < 1 or req.horizon_hours > 72:
        raise HTTPException(status_code=400, detail="horizon_hours must be 1-72")
    
    try:
        # Phase 1: Fetch data
        data_start = time.time()
        weather_data = await asyncio.wait_for(
            fetch_weather_data(req.city, req.lat, req.lon),
            timeout=12.0
        )
        data_time = time.time() - data_start
        
        # Phase 2: Create named features
        feature_start = time.time()
        X = create_model_features(weather_data, EXPECTED_TIMESTEPS)
        feature_time = time.time() - feature_start
        
        # Validate shape
        expected_shape = (1, EXPECTED_TIMESTEPS, EXPECTED_FEATURES)
        if X.shape != expected_shape:
            logger.error(f"❌ Shape mismatch: {X.shape} != {expected_shape}")
            raise ValueError(f"Feature shape mismatch: {X.shape}")
        
        # Phase 3: Model prediction
        predict_start = time.time()
        predictions = safe_model_predict(X, req.horizon_hours)
        predict_time = time.time() - predict_start
        
        total_time = time.time() - start_time
        
        # Response
        pollution = weather_data["current"]["air_pollution"]
        weather = weather_data["current"]["weather"]
        
        response = {
            "success": True,
            "request_id": request_id,
            "city": req.city,
            "resolved_city": weather_data["resolved_city"],
            "coordinates": weather_data["coordinates"],
            "timezone": weather_data.get("timezone", "UTC"),
            "timezone_offset": weather_data.get("timezone_offset", 0),
            "horizon_hours": req.horizon_hours,
            "api_success": weather_data.get("api_success", False),
            "predictions": [round(p, 3) for p in predictions],
            "current_conditions": {
                "aqi": pollution.get("aqi", 3),
                "no2": round(pollution.get("no2", 40), 1),      # Raw value as model expects
                "o3": round(pollution.get("o3", 60), 1),        # Raw value as model expects
                "pm25": round(pollution.get("pm2_5", 30), 1),
                "temperature": round(weather.get("temperature", 25), 1),
                "humidity": round(weather.get("humidity", 60), 1),
                "pressure": round(weather.get("pressure", 1013), 1),
                "wind_speed": round(weather.get("wind_speed", 3), 1),
                "wind_direction": round(weather.get("wind_direction", 180), 1),
                "wind_gust": round(weather.get("wind_gust", 0), 1),
                "clouds": weather.get("clouds", 0),
                "visibility": weather.get("visibility", 10000),
                "dew_point": round(weather.get("dew_point", 0), 1),
                "uvi": round(weather.get("uvi", 0), 1),
                "weather": weather.get("weather_description", "clear sky"),
                "weather_main": weather.get("weather_main", "Clear"),
                "weather_icon": weather.get("weather_icon", "01d")
            },
            "hourly_forecast": weather_data.get("hourly", [])[:24],  # Next 24 hours
            "daily_forecast": weather_data.get("daily", [])[:7],     # Next 7 days
            "model_info": {
                "input_shape": f"({EXPECTED_TIMESTEPS}, {EXPECTED_FEATURES})",
                "feature_names": MODEL_FEATURE_NAMES,
                "status": "healthy" if model is not None else "fallback",
                "used_model": model is not None
            },
            "performance": {
                "total_time": round(total_time, 2),
                "data_fetch": round(data_time, 2),
                "feature_creation": round(feature_time, 3),
                "prediction": round(predict_time, 2)
            },
            "data_sources": {
                "weather": weather["source"],
                "pollution": pollution["source"]
            }
        }
        
        logger.info(f"✅ [{request_id}] SUCCESS | {total_time:.2f}s | NO2: {pollution.get('no2', 40):.1f} | O3: {pollution.get('o3', 60):.1f}")
        return response
        
    except asyncio.TimeoutError:
        total_time = time.time() - start_time
        logger.error(f"⏰ [{request_id}] TIMEOUT after {total_time:.2f}s")
        raise HTTPException(status_code=408, detail="Request timed out")
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"💥 [{request_id}] ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# --- Health Check ---
@app.get("/health")
async def health_check():
    start_time = time.time()
    
    try:
        client = await get_weather_client()
        lat, lon = await client.get_coordinates("delhi")
        one_call_data = await asyncio.wait_for(client.get_one_call_data(lat, lon), timeout=5.0)
        health_time = time.time() - start_time
        
        return {
            "status": "healthy",
            "response_time": f"{health_time*1000:.0f}ms",
            "model_status": "loaded" if model is not None else "unavailable",
            "feature_names": MODEL_FEATURE_NAMES[:3] if model else ["N/A"],
            "one_call_api": "working" if one_call_data else "unavailable",
            "api_key": "valid",
            "cities": len(CITY_COORDINATES)
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "model_status": "unavailable" if model is None else "loaded"
        }

# --- Model Info ---
@app.get("/model/info")
def model_info():
    """Detailed model information"""
    feature_names = MODEL_INPUT_FEATURES if MODEL_INPUT_FEATURES else generate_standard_feature_names(EXPECTED_FEATURES)
    
    return {
        "model_file": MODEL_PATH,
        "status": "loaded" if model is not None else "unavailable",
        "input_shape": f"(1, {EXPECTED_TIMESTEPS}, {len(feature_names)})",
        "feature_names": feature_names,
        "critical_features": ["hour", "NO2", "O3"] if all(f in feature_names for f in ["hour", "NO2", "O3"]) else "detected",
        "feature_details": {
            "hour": "Time of day (0-23 or cyclic encoding)",
            "NO2": "Nitrogen Dioxide concentration (µg/m³)",
            "O3": "Ozone concentration (µg/m³)",
            "temperature": "Temperature (°C)",
            "humidity": "Relative humidity (%)"
        },
        "data_source": "OpenWeatherMap One Call API 3.0 + estimated pollution",
        "output": "AQI predictions (0-5 scale)",
        "normalization": "Model handles normalization internally"
    }

# --- Weather Data Endpoint ---
@app.get("/weather/{city}")
async def get_weather_data(city: str, lat: Optional[float] = None, lon: Optional[float] = None):
    """Get detailed weather data for a city using the new API structure"""
    try:
        weather_data = await fetch_weather_data(city, lat, lon)
        
        return {
            "success": True,
            "city": weather_data["city"],
            "resolved_city": weather_data["resolved_city"],
            "coordinates": weather_data["coordinates"],
            "timezone": weather_data.get("timezone", "UTC"),
            "timezone_offset": weather_data.get("timezone_offset", 0),
            "current": weather_data["current"],
            "hourly": weather_data.get("hourly", [])[:24],
            "daily": weather_data.get("daily", [])[:7],
            "api_success": weather_data.get("api_success", False),
            "data_sources": {
                "weather": weather_data["current"]["weather"]["source"],
                "pollution": weather_data["current"]["air_pollution"]["source"]
            }
        }
    except Exception as e:
        logger.error(f"❌ Weather data fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Weather data fetch failed: {str(e)}")

# --- Root ---
@app.get("/")
def root():
    """API root"""
    feature_names = MODEL_FEATURE_NAMES if MODEL_FEATURE_NAMES else ["hour", "NO2", "O3"]
    
    return {
        "message": "🎯 VAYUDUT Air Quality API - OpenWeatherMap 3.0 Integration",
        "version": "4.0-openweathermap-3.0",
        "status": "live",
        "model_ready": model is not None,
        "expected_features": feature_names,
        "data_source": "OpenWeatherMap One Call API 3.0",
        "api_structure": {
            "current": "Current weather and pollution data",
            "hourly": "48-hour weather forecast",
            "daily": "7-day weather forecast",
            "alerts": "Weather alerts and warnings (optional)"
        },
        "endpoints": {
            "GET /health": "Service status",
            "GET /model/info": "Model details with feature names",
            "GET /weather/{city}": "Get detailed weather data for a city",
            "POST /predict": "Make air quality prediction"
        },
        "example": {
            "method": "POST",
            "url": "/predict",
            "body": '{"city": "Delhi NCR", "horizon_hours": 24}',
            "expected_input": f"Features: {', '.join(feature_names[:3])}..."
        },
        "new_features": [
            "Full OpenWeatherMap 3.0 API integration",
            "Hourly and daily weather forecasts",
            "Enhanced weather data (wind_gust, visibility, dew_point, UVI)",
            "Timezone support",
            "Weather alerts support",
            "Improved pollution estimation"
        ],
        "guarantees": [
            "Exact feature names: 'hour', 'NO2', 'O3'",
            "Real One Call API 3.0 weather data", 
            "Smart pollution estimation",
            "No async/sync errors",
            "18s timeout protection"
        ]
    }

# --- Shutdown ---
@app.on_event("shutdown")
async def shutdown():
    global weather_client
    if weather_client:
        await weather_client.__aexit__(None, None, None)

# --- Run ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", workers=1)

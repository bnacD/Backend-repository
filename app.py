# ============ IMPORTS ============
import os
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS

import time
from collections import defaultdict, deque
import traceback

try:
    from meteostat import Point, Daily
    METEOSTAT_AVAILABLE = True
except ImportError:
    METEOSTAT_AVAILABLE = False
    print("⚠️ Meteostat not available")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available")

from functools import wraps
import hashlib
import pickle
import logging
import threading

# ============ CONFIGURATION ============
app = Flask(__name__)

def get_allowed_origins():
    """Obtener orígenes permitidos según el ambiente"""
    env = os.environ.get("FLASK_ENV", "development").lower()
    
    if env == "production":
        return [
            "https://tu-frontend-domain.com",
            "https://www.tu-frontend-domain.com",
            "https://tu-app.herokuapp.com"
        ]
    elif env == "staging":
        return [
            "https://staging.tu-frontend-domain.com",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
    else:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5500",
            "http://127.0.0.1:5500"
        ]

# CORS configuración segura
CORS(app, resources={
    r"/api/*": {
        "origins": get_allowed_origins(),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type", 
            "Authorization", 
            "X-Requested-With",
            "Accept",
            "Origin"
        ],
        "expose_headers": [
            "X-Total-Count", 
            "X-API-Version",
            "X-Rate-Limit-Remaining"
        ],
        "supports_credentials": False,
        "max_age": 86400
    }
})

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
VISUALCROSSING_KEY = os.environ.get("VISUALCROSSING_KEY", "")
CACHE_TTL_HOURS = int(os.environ.get("CACHE_TTL_HOURS", "1"))
CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")

# Variable limits para validación y detección de outliers
VARIABLE_LIMITS = {
    "Temperature (°C)": (-50, 60, 30),
    "Precipitation (mm)": (0, 500, 100),
    "Wind speed (m/s)": (0, 50, 20),
    "Humidity (%)": (0, 100, 40)
}

# ============ RATE LIMITER ============
class SimpleRateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id, limit=10, window=60):
        with self.lock:
            now = time.time()
            while self.requests[client_id] and self.requests[client_id][0] < now - window:
                self.requests[client_id].popleft()
            
            if len(self.requests[client_id]) >= limit:
                return False
            
            self.requests[client_id].append(now)
            return True
    
    def get_remaining_requests(self, client_id, limit=10):
        with self.lock:
            return max(0, limit - len(self.requests[client_id]))

rate_limiter = SimpleRateLimiter()

# ============ DATA CACHE MEJORADO ============
class DataCache:
    def __init__(self, cache_dir=CACHE_DIR, ttl_hours=CACHE_TTL_HOURS):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Verificar permisos de escritura
            test_file = os.path.join(cache_dir, 'test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"💾 Cache directory ready: {cache_dir}")
        except Exception as e:
            logger.warning(f"Cache directory issue: {e}")
            self.cache_dir = None
    
    def _get_cache_key(self, *args):
        key_string = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key):
        if not self.cache_dir:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if not os.path.exists(cache_file):
            return None
        
        try:
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.datetime.now() - file_time > datetime.timedelta(hours=self.ttl_hours):
                os.remove(cache_file)
                return None
            
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error for key {key[:8]}: {e}")
            return None
    
    def set(self, key, data):
        if not self.cache_dir:
            return False
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.warning(f"Cache write error for key {key[:8]}: {e}")
            return False
    
    def clear_expired(self):
        """Limpiar cache expirado"""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return 0
        
        cleared = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if datetime.datetime.now() - file_time > datetime.timedelta(hours=self.ttl_hours):
                        os.remove(file_path)
                        cleared += 1
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
        
        return cleared

data_cache = DataCache()

# ============ VALIDATION FUNCTIONS ============
def validate_date_parameter(date_str, param_name="date"):
    """Validar parámetro de fecha de forma estricta"""
    if not date_str:
        raise ValueError(f"Missing required parameter: {param_name}")
    
    try:
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid {param_name} format. Expected YYYY-MM-DD, got: {date_str}")
    
    min_date = datetime.date(1950, 1, 1)
    max_date = datetime.date.today() + datetime.timedelta(days=730)
    
    if date_obj < min_date:
        raise ValueError(f"{param_name} too far in the past. Minimum: {min_date}, provided: {date_obj}")
    
    if date_obj > max_date:
        raise ValueError(f"{param_name} too far in the future. Maximum: {max_date}, provided: {date_obj}")
    
    return date_obj

def validate_coordinate_parameters(lat_str, lon_str):
    """Validar coordenadas de forma estricta"""
    try:
        lat = float(lat_str) if lat_str else None
        lon = float(lon_str) if lon_str else None
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude must be valid numbers")
    
    if lat is None or lon is None:
        raise ValueError("Both latitude and longitude are required")
    
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude must be between -90 and 90. Provided: {lat}")
    
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude must be between -180 and 180. Provided: {lon}")
    
    return lat, lon

def validate_years_parameter(years_str):
    """Validar parámetro de años"""
    try:
        years = int(years_str) if years_str else 10
    except (TypeError, ValueError):
        raise ValueError("Years parameter must be a valid integer")
    
    if not (1 <= years <= 30):
        raise ValueError(f"Years must be between 1 and 30. Provided: {years}")
    
    return years

def validate_days_parameter(days_str):
    """Validar parámetro de días"""
    try:
        days = int(days_str) if days_str else 7
    except (TypeError, ValueError):
        raise ValueError("Days parameter must be a valid integer")
    
    if not (1 <= days <= 16):
        raise ValueError(f"Days must be between 1 and 16. Provided: {days}")
    
    return days

# ============ DECORATORS MEJORADOS ============
def safe_api_call(operation_name="API operation"):
    """Decorator para manejo consistente de errores en endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                logger.warning(f"🚫 {operation_name} - Validation error: {e}")
                return jsonify({
                    "error": "Invalid parameters",
                    "message": str(e),
                    "operation": operation_name
                }), 400
            except requests.RequestException as e:
                logger.error(f"🌐 {operation_name} - Network error: {e}")
                return jsonify({
                    "error": "External service unavailable",
                    "message": "Unable to fetch data from external weather services",
                    "operation": operation_name,
                    "retry_suggested": True
                }), 503
            except FileNotFoundError as e:
                logger.error(f"📁 {operation_name} - File error: {e}")
                return jsonify({
                    "error": "Resource not found",
                    "message": "Required resource or cache file not found",
                    "operation": operation_name
                }), 404
            except Exception as e:
                error_id = f"ERR_{int(time.time())}"
                logger.error(f"💥 {operation_name} - Unexpected error [{error_id}]: {e}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                
                return jsonify({
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "error_id": error_id,
                    "operation": operation_name,
                    "contact": "Please report this error ID for support"
                }), 500
        return decorated_function
    return decorator

def rate_limit(limit=10):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = request.remote_addr
            if not rate_limiter.is_allowed(client_id, limit):
                remaining = rate_limiter.get_remaining_requests(client_id, limit)
                return jsonify({
                    "error": "Rate limit exceeded",
                    "remaining_requests": remaining,
                    "retry_after": 60
                }), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_request_params(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            lat_str = request.args.get("lat")
            lon_str = request.args.get("lon")
            lat, lon = validate_coordinate_parameters(lat_str, lon_str)
            
            variable = request.args.get("variable", "Temperature (°C)")
            if variable not in VARIABLE_LIMITS:
                raise ValueError(f"Invalid variable '{variable}'. Allowed: {list(VARIABLE_LIMITS.keys())}")
            
            type_ = request.args.get("type", "").lower()
            if type_ and type_ not in ["real", "ia"]:
                raise ValueError(f"Invalid type '{type_}'. Must be 'real' or 'ia'")
            
            if type_ == "ia":
                date_str = request.args.get("date")
                validate_date_parameter(date_str, "date")
                
                years_str = request.args.get("years")
                if years_str:
                    validate_years_parameter(years_str)
            
            elif type_ == "real":
                days_str = request.args.get("days")
                if days_str:
                    validate_days_parameter(days_str)
            
            return f(*args, **kwargs)
            
        except ValueError as e:
            logger.warning(f"🚫 Parameter validation error: {e}")
            return jsonify({
                "error": "Parameter validation failed",
                "message": str(e),
                "help": "Check the API documentation for correct parameter formats"
            }), 400
        except Exception as e:
            logger.error(f"❌ Unexpected validation error: {e}")
            return jsonify({
                "error": "Validation error",
                "message": "An unexpected error occurred during parameter validation"
            }), 500
    
    return decorated_function

# ============ MIDDLEWARE ============
@app.before_request
def log_request():
    if hasattr(request, 'path') and request.path.startswith('/api/'):
        if np.random.random() < 0.01:
            cleared = data_cache.clear_expired()
            if cleared > 0:
                logger.info(f"🧹 Cleared {cleared} expired cache files")
    
    logger.info(f"📥 {request.method} {request.url} from {request.remote_addr}")

@app.after_request
def log_response(response):
    response.headers["X-API-Version"] = "3.1"
    response.headers["X-Powered-By"] = "NASA-Paraguay Weather API"
    response.headers["Cache-Control"] = "public, max-age=300"
    
    if os.environ.get("FLASK_ENV") == "production":
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    logger.info(f"📤 Response: {response.status_code}")
    return response

# ============ UTILITY FUNCTIONS ============
def clean_invalid_values(df):
    """Limpiar valores inválidos del dataframe"""
    if df.empty:
        return df
    
    original_count = len(df)
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    df = df[np.isfinite(df['value'])]
    df = df[df['value'] != -999]
    df = df[df['value'] != -9999]
    
    cleaned_count = len(df)
    if original_count != cleaned_count:
        logger.debug(f"🧹 Cleaned {original_count - cleaned_count} invalid values")
    
    return df

def remove_outliers(df, variable):
    """Remover outliers estadísticos basados en límites de variables"""
    if df.empty or len(df) < 10:
        return df
    
    original_count = len(df)
    df = clean_invalid_values(df)
    if df.empty:
        return df
    
    min_limit, max_limit, _ = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
    df = df[(df['value'] >= min_limit) & (df['value'] <= max_limit)]
    
    if len(df) < 5:
        return df
    
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR
    
    df_filtered = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    
    outliers_removed = len(df) - len(df_filtered)
    if outliers_removed > 0:
        logger.debug(f"🎯 Removed {outliers_removed} outliers for {variable}")
    
    return df_filtered

def validate_date_range(start_date, end_date, max_days=365*5):
    """Validar rango de fechas"""
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")
    
    days_diff = (end_date - start_date).days
    if days_diff > max_days:
        raise ValueError(f"Date range too large: {days_diff} days (max: {max_days})")
    
    return start_date, end_date

# ============ DATA FETCHING FUNCTIONS ============
def fetch_metar(icao_code="SGAS"):
    """Obtener datos METAR con manejo de errores mejorado"""
    operation = f"METAR fetch for {icao_code}"
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{icao_code}.TXT"
        
        logger.debug(f"🛫 {operation}: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            logger.warning(f"⚠️ {operation}: Insufficient data in response")
            return None
        
        fecha_obs = lines[0]
        metar_data = lines[1]
        
        result = {
            "icao": icao_code,
            "fecha_obs": fecha_obs,
            "metar_raw": metar_data,
            "source": "NOAA METAR"
        }
        
        # Parse METAR data
        parts = metar_data.split()
        for part in parts:
            if part.endswith("KT") and len(part) >= 5:
                try:
                    wind_speed_kt = int(part[3:5])
                    result["viento_mps"] = round(wind_speed_kt * 0.514444, 1)
                except ValueError:
                    pass
            
            if "/" in part and len(part) <= 7:
                try:
                    temp_str, dew_str = part.split("/")
                    
                    if temp_str.startswith("M"):
                        temp_c = -int(temp_str[1:])
                    else:
                        temp_c = int(temp_str)
                    
                    if dew_str.startswith("M"):
                        dew_c = -int(dew_str[1:])
                    else:
                        dew_c = int(dew_str)
                    
                    result["temperatura_C"] = temp_c
                    
                    if temp_c != dew_c:
                        vapor_pressure = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
                        saturation_pressure = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
                        humidity = (vapor_pressure / saturation_pressure) * 100
                        result["humedad_pct"] = round(min(100, max(0, humidity)), 1)
                except (ValueError, ZeroDivisionError):
                    pass
        
        logger.info(f"✅ {operation}: Success")
        return result
        
    except requests.Timeout:
        logger.error(f"⏱️ {operation}: Timeout after 10 seconds")
        return None
    except requests.ConnectionError:
        logger.error(f"🔌 {operation}: Connection failed")
        return None
    except requests.HTTPError as e:
        logger.error(f"🌐 {operation}: HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"❌ {operation}: Unexpected error - {e}")
        return None

def fetch_hourly_openmeteo(lat, lon, variable, days=7):
    """Obtener datos horarios de Open-Meteo"""
    try:
        var_mapping = {
            "Temperature (°C)": "temperature_2m",
            "Precipitation (mm)": "precipitation",
            "Wind speed (m/s)": "windspeed_10m",
            "Humidity (%)": "relativehumidity_2m"
        }
        
        param = var_mapping.get(variable)
        if not param:
            logger.error(f"Invalid variable for Open-Meteo: {variable}")
            return [], {}
        
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly={param}"
            f"&forecast_days={min(days, 16)}&timezone=auto"
        )
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        values = hourly.get(param, [])
        
        if not times or not values:
            logger.warning("Empty hourly data from Open-Meteo")
            return [], {}
        
        records = []
        for time_str, value in zip(times, values):
            if value is not None and not pd.isna(value):
                dt = pd.to_datetime(time_str)
                records.append({
                    "date": dt.isoformat(),
                    "value": float(value)
                })
        
        metadata = {
            "source": "Open-Meteo",
            "location": {"latitude": lat, "longitude": lon},
            "variable": variable,
            "forecast_days": days,
            "timezone": data.get("timezone", "UTC"),
            "total_records": len(records)
        }
        
        logger.info(f"✅ Open-Meteo: {len(records)} records fetched")
        return records, metadata
        
    except Exception as e:
        logger.error(f"Open-Meteo fetch error: {e}")
        return [], {}

def fetch_meteostat_daily(lat, lon, start, end, variable):
    """Obtener datos diarios históricos de Meteostat"""
    if not METEOSTAT_AVAILABLE:
        logger.warning("Meteostat not available, skipping")
        return pd.DataFrame()
    
    column_mapping = {
        "Temperature (°C)": "tavg",
        "Precipitation (mm)": "prcp", 
        "Wind speed (m/s)": "wspd",
        "Humidity (%)": "rhum"
    }
    
    column = column_mapping.get(variable)
    if not column:
        logger.error(f"Invalid variable for Meteostat: {variable}")
        return pd.DataFrame()
    
    try:
        start_date, end_date = validate_date_range(start, end)
        
        station_point = Point(lat, lon)
        
        if isinstance(start_date, datetime.date) and not isinstance(start_date, datetime.datetime):
            start_dt = datetime.datetime.combine(start_date, datetime.time.min)
        else:
            start_dt = start_date
            
        if isinstance(end_date, datetime.date) and not isinstance(end_date, datetime.datetime):
            end_dt = datetime.datetime.combine(end_date, datetime.time.max)
        else:
            end_dt = end_date
        
        logger.info(f"🌍 Fetching Meteostat data from {start_dt.date()} to {end_dt.date()}")
        
        df = Daily(station_point, start=start_dt, end=end_dt).fetch()
        
        if df.empty or column not in df.columns:
            logger.warning(f"No Meteostat data available for {variable}")
            return pd.DataFrame()
        
        df['value'] = df[column]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.dropna(subset=['value'], inplace=True)
        
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        logger.info(f"✅ Meteostat: {len(df)} records fetched")
        return clean_invalid_values(df)
        
    except Exception as e:
        logger.error(f"Meteostat daily fetch error: {e}")
        return pd.DataFrame()

def fetch_nasa_daily(lat, lon, start, end, variable):
    """Obtener datos diarios de NASA POWER"""
    var_map = {
        "Temperature (°C)": "T2M",
        "Precipitation (mm)": "PRECTOTCORR",
        "Wind speed (m/s)": "WS10M", 
        "Humidity (%)": "RH2M"
    }
    
    param = var_map.get(variable)
    if not param:
        logger.error(f"Invalid variable for NASA POWER: {variable}")
        return pd.DataFrame()
    
    try:
        ## 🔧 **Código Backend Completo - Versión Sin Errores**
# ============ IMPORTS ============
import os
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS

import time
from collections import defaultdict, deque
import traceback

try:
    from meteostat import Point, Daily
    METEOSTAT_AVAILABLE = True
except ImportError:
    METEOSTAT_AVAILABLE = False
    print("⚠️ Meteostat not available")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available")

from functools import wraps
import hashlib
import pickle
import logging
import threading

# ============ CONFIGURATION ============
app = Flask(__name__)

def get_allowed_origins():
    """Obtener orígenes permitidos según el ambiente"""
    env = os.environ.get("FLASK_ENV", "development").lower()
    
    if env == "production":
        return [
            "https://tu-frontend-domain.com",
            "https://www.tu-frontend-domain.com",
            "https://tu-app.herokuapp.com"
        ]
    elif env == "staging":
        return [
            "https://staging.tu-frontend-domain.com",
            "http://localhost:3000",
            "http://127.0.0.1:3000"
        ]
    else:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5500",
            "http://127.0.0.1:5500"
        ]

# CORS configuración segura
CORS(app, resources={
    r"/api/*": {
        "origins": get_allowed_origins(),
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type", 
            "Authorization", 
            "X-Requested-With",
            "Accept",
            "Origin"
        ],
        "expose_headers": [
            "X-Total-Count", 
            "X-API-Version",
            "X-Rate-Limit-Remaining"
        ],
        "supports_credentials": False,
        "max_age": 86400
    }
})

# Logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
VISUALCROSSING_KEY = os.environ.get("VISUALCROSSING_KEY", "")
CACHE_TTL_HOURS = int(os.environ.get("CACHE_TTL_HOURS", "1"))
CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")

# Variable limits para validación y detección de outliers
VARIABLE_LIMITS = {
    "Temperature (°C)": (-50, 60, 30),
    "Precipitation (mm)": (0, 500, 100),
    "Wind speed (m/s)": (0, 50, 20),
    "Humidity (%)": (0, 100, 40)
}

# ============ RATE LIMITER ============
class SimpleRateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id, limit=10, window=60):
        with self.lock:
            now = time.time()
            while self.requests[client_id] and self.requests[client_id][0] < now - window:
                self.requests[client_id].popleft()
            
            if len(self.requests[client_id]) >= limit:
                return False
            
            self.requests[client_id].append(now)
            return True
    
    def get_remaining_requests(self, client_id, limit=10):
        with self.lock:
            return max(0, limit - len(self.requests[client_id]))

rate_limiter = SimpleRateLimiter()

# ============ DATA CACHE MEJORADO ============
class DataCache:
    def __init__(self, cache_dir=CACHE_DIR, ttl_hours=CACHE_TTL_HOURS):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Verificar permisos de escritura
            test_file = os.path.join(cache_dir, 'test.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info(f"💾 Cache directory ready: {cache_dir}")
        except Exception as e:
            logger.warning(f"Cache directory issue: {e}")
            self.cache_dir = None
    
    def _get_cache_key(self, *args):
        key_string = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key):
        if not self.cache_dir:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if not os.path.exists(cache_file):
            return None
        
        try:
            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.datetime.now() - file_time > datetime.timedelta(hours=self.ttl_hours):
                os.remove(cache_file)
                return None
            
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error for key {key[:8]}: {e}")
            return None
    
    def set(self, key, data):
        if not self.cache_dir:
            return False
        
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.warning(f"Cache write error for key {key[:8]}: {e}")
            return False
    
    def clear_expired(self):
        """Limpiar cache expirado"""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return 0
        
        cleared = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                    if datetime.datetime.now() - file_time > datetime.timedelta(hours=self.ttl_hours):
                        os.remove(file_path)
                        cleared += 1
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
        
        return cleared

data_cache = DataCache()

# ============ VALIDATION FUNCTIONS ============
def validate_date_parameter(date_str, param_name="date"):
    """Validar parámetro de fecha de forma estricta"""
    if not date_str:
        raise ValueError(f"Missing required parameter: {param_name}")
    
    try:
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid {param_name} format. Expected YYYY-MM-DD, got: {date_str}")
    
    min_date = datetime.date(1950, 1, 1)
    max_date = datetime.date.today() + datetime.timedelta(days=730)
    
    if date_obj < min_date:
        raise ValueError(f"{param_name} too far in the past. Minimum: {min_date}, provided: {date_obj}")
    
    if date_obj > max_date:
        raise ValueError(f"{param_name} too far in the future. Maximum: {max_date}, provided: {date_obj}")
    
    return date_obj

def validate_coordinate_parameters(lat_str, lon_str):
    """Validar coordenadas de forma estricta"""
    try:
        lat = float(lat_str) if lat_str else None
        lon = float(lon_str) if lon_str else None
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude must be valid numbers")
    
    if lat is None or lon is None:
        raise ValueError("Both latitude and longitude are required")
    
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude must be between -90 and 90. Provided: {lat}")
    
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude must be between -180 and 180. Provided: {lon}")
    
    return lat, lon

def validate_years_parameter(years_str):
    """Validar parámetro de años"""
    try:
        years = int(years_str) if years_str else 10
    except (TypeError, ValueError):
        raise ValueError("Years parameter must be a valid integer")
    
    if not (1 <= years <= 30):
        raise ValueError(f"Years must be between 1 and 30. Provided: {years}")
    
    return years

def validate_days_parameter(days_str):
    """Validar parámetro de días"""
    try:
        days = int(days_str) if days_str else 7
    except (TypeError, ValueError):
        raise ValueError("Days parameter must be a valid integer")
    
    if not (1 <= days <= 16):
        raise ValueError(f"Days must be between 1 and 16. Provided: {days}")
    
    return days

# ============ DECORATORS MEJORADOS ============
def safe_api_call(operation_name="API operation"):
    """Decorator para manejo consistente de errores en endpoints"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ValueError as e:
                logger.warning(f"🚫 {operation_name} - Validation error: {e}")
                return jsonify({
                    "error": "Invalid parameters",
                    "message": str(e),
                    "operation": operation_name
                }), 400
            except requests.RequestException as e:
                logger.error(f"🌐 {operation_name} - Network error: {e}")
                return jsonify({
                    "error": "External service unavailable",
                    "message": "Unable to fetch data from external weather services",
                    "operation": operation_name,
                    "retry_suggested": True
                }), 503
            except FileNotFoundError as e:
                logger.error(f"📁 {operation_name} - File error: {e}")
                return jsonify({
                    "error": "Resource not found",
                    "message": "Required resource or cache file not found",
                    "operation": operation_name
                }), 404
            except Exception as e:
                error_id = f"ERR_{int(time.time())}"
                logger.error(f"💥 {operation_name} - Unexpected error [{error_id}]: {e}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                
                return jsonify({
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "error_id": error_id,
                    "operation": operation_name,
                    "contact": "Please report this error ID for support"
                }), 500
        return decorated_function
    return decorator

def rate_limit(limit=10):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = request.remote_addr
            if not rate_limiter.is_allowed(client_id, limit):
                remaining = rate_limiter.get_remaining_requests(client_id, limit)
                return jsonify({
                    "error": "Rate limit exceeded",
                    "remaining_requests": remaining,
                    "retry_after": 60
                }), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_request_params(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            lat_str = request.args.get("lat")
            lon_str = request.args.get("lon")
            lat, lon = validate_coordinate_parameters(lat_str, lon_str)
            
            variable = request.args.get("variable", "Temperature (°C)")
            if variable not in VARIABLE_LIMITS:
                raise ValueError(f"Invalid variable '{variable}'. Allowed: {list(VARIABLE_LIMITS.keys())}")
            
            type_ = request.args.get("type", "").lower()
            if type_ and type_ not in ["real", "ia"]:
                raise ValueError(f"Invalid type '{type_}'. Must be 'real' or 'ia'")
            
            if type_ == "ia":
                date_str = request.args.get("date")
                validate_date_parameter(date_str, "date")
                
                years_str = request.args.get("years")
                if years_str:
                    validate_years_parameter(years_str)
            
            elif type_ == "real":
                days_str = request.args.get("days")
                if days_str:
                    validate_days_parameter(days_str)
            
            return f(*args, **kwargs)
            
        except ValueError as e:
            logger.warning(f"🚫 Parameter validation error: {e}")
            return jsonify({
                "error": "Parameter validation failed",
                "message": str(e),
                "help": "Check the API documentation for correct parameter formats"
            }), 400
        except Exception as e:
            logger.error(f"❌ Unexpected validation error: {e}")
            return jsonify({
                "error": "Validation error",
                "message": "An unexpected error occurred during parameter validation"
            }), 500
    
    return decorated_function

# ============ MIDDLEWARE ============
@app.before_request
def log_request():
    if hasattr(request, 'path') and request.path.startswith('/api/'):
        if np.random.random() < 0.01:
            cleared = data_cache.clear_expired()
            if cleared > 0:
                logger.info(f"🧹 Cleared {cleared} expired cache files")
    
    logger.info(f"📥 {request.method} {request.url} from {request.remote_addr}")

@app.after_request
def log_response(response):
    response.headers["X-API-Version"] = "3.1"
    response.headers["X-Powered-By"] = "NASA-Paraguay Weather API"
    response.headers["Cache-Control"] = "public, max-age=300"
    
    if os.environ.get("FLASK_ENV") == "production":
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    logger.info(f"📤 Response: {response.status_code}")
    return response

# ============ UTILITY FUNCTIONS ============
def clean_invalid_values(df):
    """Limpiar valores inválidos del dataframe"""
    if df.empty:
        return df
    
    original_count = len(df)
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    df = df[np.isfinite(df['value'])]
    df = df[df['value'] != -999]
    df = df[df['value'] != -9999]
    
    cleaned_count = len(df)
    if original_count != cleaned_count:
        logger.debug(f"🧹 Cleaned {original_count - cleaned_count} invalid values")
    
    return df

def remove_outliers(df, variable):
    """Remover outliers estadísticos basados en límites de variables"""
    if df.empty or len(df) < 10:
        return df
    
    original_count = len(df)
    df = clean_invalid_values(df)
    if df.empty:
        return df
    
    min_limit, max_limit, _ = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
    df = df[(df['value'] >= min_limit) & (df['value'] <= max_limit)]
    
    if len(df) < 5:
        return df
    
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2.5 * IQR
    upper_bound = Q3 + 2.5 * IQR
    
    df_filtered = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    
    outliers_removed = len(df) - len(df_filtered)
    if outliers_removed > 0:
        logger.debug(f"🎯 Removed {outliers_removed} outliers for {variable}")
    
    return df_filtered

def validate_date_range(start_date, end_date, max_days=365*5):
    """Validar rango de fechas"""
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    if start_date > end_date:
        raise ValueError("Start date cannot be after end date")
    
    days_diff = (end_date - start_date).days
    if days_diff > max_days:
        raise ValueError(f"Date range too large: {days_diff} days (max: {max_days})")
    
    return start_date, end_date

# ============ DATA FETCHING FUNCTIONS ============
def fetch_metar(icao_code="SGAS"):
    """Obtener datos METAR con manejo de errores mejorado"""
    operation = f"METAR fetch for {icao_code}"
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{icao_code}.TXT"
        
        logger.debug(f"🛫 {operation}: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            logger.warning(f"⚠️ {operation}: Insufficient data in response")
            return None
        
        fecha_obs = lines[0]
        metar_data = lines[1]
        
        result = {
            "icao": icao_code,
            "fecha_obs": fecha_obs,
            "metar_raw": metar_data,
            "source": "NOAA METAR"
        }
        
        # Parse METAR data
        parts = metar_data.split()
        for part in parts:
            if part.endswith("KT") and len(part) >= 5:
                try:
                    wind_speed_kt = int(part[3:5])
                    result["viento_mps"] = round(wind_speed_kt * 0.514444, 1)
                except ValueError:
                    pass
            
            if "/" in part and len(part) <= 7:
                try:
                    temp_str, dew_str = part.split("/")
                    
                    if temp_str.startswith("M"):
                        temp_c = -int(temp_str[1:])
                    else:
                        temp_c = int(temp_str)
                    
                    if dew_str.startswith("M"):
                        dew_c = -int(dew_str[1:])
                    else:
                        dew_c = int(dew_str)
                    
                    result["temperatura_C"] = temp_c
                    
                    if temp_c != dew_c:
                        vapor_pressure = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
                        saturation_pressure = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
                        humidity = (vapor_pressure / saturation_pressure) * 100
                        result["humedad_pct"] = round(min(100, max(0, humidity)), 1)
                except (ValueError, ZeroDivisionError):
                    pass
        
        logger.info(f"✅ {operation}: Success")
        return result
        
    except requests.Timeout:
        logger.error(f"⏱️ {operation}: Timeout after 10 seconds")
        return None
    except requests.ConnectionError:
        logger.error(f"🔌 {operation}: Connection failed")
        return None
    except requests.HTTPError as e:
        logger.error(f"🌐 {operation}: HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"❌ {operation}: Unexpected error - {e}")
        return None

def fetch_hourly_openmeteo(lat, lon, variable, days=7):
    """Obtener datos horarios de Open-Meteo"""
    try:
        var_mapping = {
            "Temperature (°C)": "temperature_2m",
            "Precipitation (mm)": "precipitation",
            "Wind speed (m/s)": "windspeed_10m",
            "Humidity (%)": "relativehumidity_2m"
        }
        
        param = var_mapping.get(variable)
        if not param:
            logger.error(f"Invalid variable for Open-Meteo: {variable}")
            return [], {}
        
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&hourly={param}"
            f"&forecast_days={min(days, 16)}&timezone=auto"
        )
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        values = hourly.get(param, [])
        
        if not times or not values:
            logger.warning("Empty hourly data from Open-Meteo")
            return [], {}
        
        records = []
        for time_str, value in zip(times, values):
            if value is not None and not pd.isna(value):
                dt = pd.to_datetime(time_str)
                records.append({
                    "date": dt.isoformat(),
                    "value": float(value)
                })
        
        metadata = {
            "source": "Open-Meteo",
            "location": {"latitude": lat, "longitude": lon},
            "variable": variable,
            "forecast_days": days,
            "timezone": data.get("timezone", "UTC"),
            "total_records": len(records)
        }
        
        logger.info(f"✅ Open-Meteo: {len(records)} records fetched")
        return records, metadata
        
    except Exception as e:
        logger.error(f"Open-Meteo fetch error: {e}")
        return [], {}

def fetch_meteostat_daily(lat, lon, start, end, variable):
    """Obtener datos diarios históricos de Meteostat"""
    if not METEOSTAT_AVAILABLE:
        logger.warning("Meteostat not available, skipping")
        return pd.DataFrame()
    
    column_mapping = {
        "Temperature (°C)": "tavg",
        "Precipitation (mm)": "prcp", 
        "Wind speed (m/s)": "wspd",
        "Humidity (%)": "rhum"
    }
    
    column = column_mapping.get(variable)
    if not column:
        logger.error(f"Invalid variable for Meteostat: {variable}")
        return pd.DataFrame()
    
    try:
        start_date, end_date = validate_date_range(start, end)
        
        station_point = Point(lat, lon)
        
        if isinstance(start_date, datetime.date) and not isinstance(start_date, datetime.datetime):
            start_dt = datetime.datetime.combine(start_date, datetime.time.min)
        else:
            start_dt = start_date
            
        if isinstance(end_date, datetime.date) and not isinstance(end_date, datetime.datetime):
            end_dt = datetime.datetime.combine(end_date, datetime.time.max)
        else:
            end_dt = end_date
        
        logger.info(f"🌍 Fetching Meteostat data from {start_dt.date()} to {end_dt.date()}")
        
        df = Daily(station_point, start=start_dt, end=end_dt).fetch()
        
        if df.empty or column not in df.columns:
            logger.warning(f"No Meteostat data available for {variable}")
            return pd.DataFrame()
        
        df['value'] = df[column]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.dropna(subset=['value'], inplace=True)
        
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        logger.info(f"✅ Meteostat: {len(df)} records fetched")
        return clean_invalid_values(df)
        
    except Exception as e:
        logger.error(f"Meteostat daily fetch error: {e}")
        return pd.DataFrame()

def fetch_nasa_daily(lat, lon, start, end, variable):
    """Obtener datos diarios de NASA POWER"""
    var_map = {
        "Temperature (°C)": "T2M",
        "Precipitation (mm)": "PRECTOTCORR",
        "Wind speed (m/s)": "WS10M", 
        "Humidity (%)": "RH2M"
    }
    
    param = var_map.get(variable)
    if not param:
        logger.error(f"Invalid variable for NASA POWER: {variable}")
        return pd.DataFrame()
    
    try:
        start_date, end_date = validate_date_range(start, end)
        
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.date()
        
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={param}&community=RE&longitude={lon}&latitude={lat}"
            f"&start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}&format=JSON"
        )
        
        logger.info(f"🚀 Fetching NASA POWER data from {start_date} to {end_date}")
        
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        
        json_data = response.json()
        data = json_data.get("properties", {}).get("parameter", {}).get(param, {})
        
        if not data:
            logger.warning("No NASA POWER data in response")
            return pd.DataFrame()
        
        records = []
        for date_str, value in data.items():
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                if start_date <= date_obj <= end_date and value is not None:
                    records.append({"date": date_obj, "value": float(value)})
            except (ValueError, TypeError) as e:
                logger.debug(f"Error parsing NASA data point {date_str}: {value}, error: {e}")
                continue
        
        logger.info(f"✅ NASA POWER: {len(records)} records fetched")
        return clean_invalid_values(pd.DataFrame(records))
        
    except Exception as e:
        logger.error(f"NASA POWER fetch error: {e}")
        return pd.DataFrame()

# ============ PREDICTION FUNCTIONS ============
def get_historical_day_average(lat, lon, target_date, variable, years_window=10):
    """Obtener promedio histórico para una fecha específica - VERSIÓN COMPLETA"""
    try:
        if not isinstance(target_date, datetime.date):
            if isinstance(target_date, str):
                target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
            else:
                raise ValueError("target_date must be a date object or YYYY-MM-DD string")
        
        if not (1 <= years_window <= 50):
            raise ValueError("years_window must be between 1 and 50")
        
        if variable not in VARIABLE_LIMITS:
            raise ValueError(f"Invalid variable: {variable}")
        
        cache_key = data_cache._get_cache_key(lat, lon, target_date, variable, f"hist{years_window}")
        cached_result = data_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"📦 Using cached historical data for {target_date}")
            return cached_result
        
        logger.info(f"🔍 Fetching historical data for {target_date} (±{years_window} years)")
        
        records = []
        successful_years = 0
        failed_years = 0
        
        for year_offset in range(years_window):
            try:
                query_year = target_date.year - year_offset - 1
                
                if query_year < 1950:
                    continue
                
                if target_date.month == 2 and target_date.day == 29:
                    try:
                        query_date = datetime.date(query_year, 2, 29)
                    except ValueError:
                        query_date = datetime.date(query_year, 2, 28)
                        logger.debug(f"Adjusted leap year date for {query_year}: Feb 29 -> Feb 28")
                else:
                    query_date = datetime.date(query_year, target_date.month, target_date.day)
                
                df = fetch_meteostat_daily(lat, lon, query_date, query_date, variable)
                if df.empty:
                    df = fetch_nasa_daily(lat, lon, query_date, query_date, variable)
                
                if not df.empty:
                    matching_rows = df[df['date'] == query_date]
                    if not matching_rows.empty:
                        value = float(matching_rows['value'].iloc[0])
                        
                        min_limit, max_limit, _ = VARIABLE_LIMITS[variable]
                        if min_limit <= value <= max_limit:
                            records.append({
                                "year": query_year,
                                "date": query_date.isoformat(),
                                "value": round(value, 2)
                            })
                            successful_years += 1
                        else:
                            logger.warning(f"Value out of range for {query_year}: {value}")
                            failed_years += 1
                    else:
                        failed_years += 1
                else:
                    failed_years += 1
                    
            except Exception as e:
                logger.warning(f"Error processing year {query_year}: {e}")
                failed_years += 1
                continue
        
        if len(records) == 0:
            logger.error(f"No historical data found for {target_date} after checking {years_window} years")
            return None
        
        if len(records) < 2:
            logger.warning(f"Limited historical data: only {len(records)} records found")
        
        values = [r['value'] for r in records]
        
        result = {
            "date": target_date.isoformat(),
            "average": round(float(np.mean(values)), 2),
            "min": round(float(min(values)), 2),
            "max": round(float(max(values)), 2),
            "std_dev": round(float(np.std(values)), 2) if len(values) > 1 else 0.0,
            "samples": len(records),
            "years_covered": sorted([r['year'] for r in records]),
            "years_data": records,
            "source": "Historical observations (Meteostat + NASA POWER)",
            "data_quality": {
                "successful_years": successful_years,
                "failed_years": failed_years,
                "success_rate": round((successful_years / (successful_years + failed_years)) * 100, 1),
                "data_span_years": max([r['year'] for r in records]) - min([r['year'] for r in records]) + 1 if records else 0
            }
        }
        
        if data_cache.set(cache_key, result):
            logger.debug(f"💾 Historical data cached successfully")
        
        logger.info(f"✅ Historical analysis completed: {len(records)} years, {result['data_quality']['success_rate']}% success rate")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in get_historical_day_average: {e}")
        return None

def simple_prediction(lat, lon, target_date, variable, years_window=5):
    """Predicción estadística simple usando promedios históricos"""
    cache_key = data_cache._get_cache_key(lat, lon, target_date, variable, f"pred{years_window}")
    cached_result = data_cache.get(cache_key)
    
    if cached_result:
        logger.info(f"📦 Using cached simple prediction for {target_date}")
        return cached_result
    
    historical_values = []
    
    for year_offset in range(1, years_window + 1):
        try:
            query_year = target_date.year - year_offset
            
            if target_date.month == 2 and target_date.day == 29:
                try:
                    query_date = datetime.date(query_year, 2, 29)
                except ValueError:
                    query_date = datetime.date(query_year, 2, 28)
            else:
                query_date = datetime.date(query_year, target_date.month, target_date.day)
            
            start_window = query_date - datetime.timedelta(days=3)
            end_window = query_date + datetime.timedelta(days=3)
            
            df = fetch_meteostat_daily(lat, lon, start_window, end_window, variable)
            if df.empty:
                df = fetch_nasa_daily(lat, lon, start_window, end_window, variable)
            
            if not df.empty:
                df = remove_outliers(df, variable)
                if not df.empty:
                    avg_value = df['value'].mean()
                    if not pd.isna(avg_value):
                        historical_values.append(avg_value)
        except Exception as e:
            logger.debug(f"Error in year {query_year}: {e}")
            continue
    
    if len(historical_values) < 2:
        logger.warning(f"Insufficient data for simple prediction: {len(historical_values)} values")
        return None
    
    mean_value = np.mean(historical_values)
    std_value = np.std(historical_values)
    
    min_limit, max_limit, _ = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
    predicted_value = np.clip(mean_value, min_limit, max_limit)
    lower_bound = np.clip(mean_value - std_value, min_limit, max_limit)
    upper_bound = np.clip(mean_value + std_value, min_limit, max_limit)
    
    result = {
        "date": target_date.isoformat(),
        "predicted_value": round(float(predicted_value), 2),
        "lower_bound": round(float(lower_bound), 2),
        "upper_bound": round(float(upper_bound), 2),
        "confidence": "Historical Average (±1σ)",
        "model": "Simple Statistical",
        "training_samples": len(historical_values),
        "training_range": {
            "start": (target_date - datetime.timedelta(days=365 * years_window)).isoformat(),
            "end": (target_date - datetime.timedelta(days=1)).isoformat()
        },
        "historical_values": [round(float(v), 2) for v in historical_values],
        "std_deviation": round(float(std_value), 2),
        "data_quality": {
            "interval_width": round(float(upper_bound - lower_bound), 2),
            "method": "statistical_average"
        }
    }
    
    data_cache.set(cache_key, result)
    logger.info(f"✅ Simple prediction completed with {len(historical_values)} historical values")
    return result

def train_predict_daily_advanced(lat, lon, target_date, variable, years_window=10):
    """Predicción avanzada usando Prophet ML model con fallback estadístico"""
    if not PROPHET_AVAILABLE:
        logger.warning("🤖 Prophet not available, using statistical fallback")
        return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
    
    try:
        start_date = target_date - datetime.timedelta(days=365 * years_window)
        end_date = target_date - datetime.timedelta(days=1)
        
        cache_key = data_cache._get_cache_key(
            lat, lon, start_date, end_date, f"{variable}_prophet_{years_window}"
        )
        cached_result = data_cache.get(cache_key)
        
        if cached_result:
            logger.info(f"📦 Using cached Prophet prediction for {target_date}")
            cached_result["date"] = target_date.isoformat()
            return cached_result
        
        logger.info(f"🤖 Training Prophet model for {variable} prediction")
        
        hist_df = fetch_meteostat_daily(lat, lon, start_date, end_date, variable)
        if hist_df.empty:
            hist_df = fetch_nasa_daily(lat, lon, start_date, end_date, variable)
        
        if hist_df.empty or len(hist_df) < 100:
            logger.warning(f"📊 Insufficient data for Prophet: {len(hist_df)} records")
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        
        original_count = len(hist_df)
        hist_df = remove_outliers(hist_df, variable)
        outliers_removed = original_count - len(hist_df)
        
        if len(hist_df) < 50:
            logger.warning(f"📊 Insufficient data after outlier removal: {len(hist_df)}")
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        
        hist_df = hist_df.copy()
        hist_df['ds'] = pd.to_datetime(hist_df['date'])
        hist_df['y'] = hist_df['value']
        hist_df = hist_df[['ds', 'y']].sort_values('ds')
        hist_df = hist_df.drop_duplicates(subset=['ds'])
        
        prophet_params = {
            "Temperature (°C)": {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'daily_seasonality': False
            },
            "Precipitation (mm)": {
                'changepoint_prior_scale': 0.5,
                'seasonality_prior_scale': 0.1,
                'yearly_seasonality': True,
                'daily_seasonality': False
            },
            "Wind speed (m/s)": {
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 1.0,
                'yearly_seasonality': True,
                'daily_seasonality': True
            },
            "Humidity (%)": {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 5.0,
                'yearly_seasonality': True,
                'daily_seasonality': False
            }
        }
        
        params = prophet_params.get(variable, prophet_params["Temperature (°C)"])
        
        model = Prophet(
            interval_width=0.80,
            seasonality_mode='additive',
            **params
        )
        
        if variable in ["Temperature (°C)", "Humidity (%)"]:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        prophet_logger = logging.getLogger('prophet')
        prophet_logger.setLevel(logging.WARNING)
        
        model.fit(hist_df)
        
        future = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
        forecast = model.predict(future)
        
        predicted_value = float(forecast['yhat'].iloc[0])
        lower_bound = float(forecast['yhat_lower'].iloc[0])
        upper_bound = float(forecast['yhat_upper'].iloc[0])
        
        min_limit, max_limit, max_interval = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
        predicted_value = np.clip(predicted_value, min_limit, max_limit)
        lower_bound = np.clip(lower_bound, min_limit, max_limit)
        upper_bound = np.clip(upper_bound, min_limit, max_limit)
        
        interval_width = upper_bound - lower_bound
        
        if interval_width > max_interval:
            logger.warning(f"⚠️ Wide prediction interval: {interval_width:.1f} > {max_interval}")
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        
        result = {
            "date": target_date.isoformat(),
            "predicted_value": round(predicted_value, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "confidence": "80%",
            "model": "Prophet (Advanced ML)",
            "training_samples": len(hist_df),
            "training_range": {
                "start": hist_df['ds'].min().date().isoformat(),
                "end": hist_df['ds'].max().date().isoformat()
            },
            "data_quality": {
                "outliers_removed": outliers_removed,
                "interval_width": round(interval_width, 2),
                "method": "prophet_ml_advanced",
                "model_params": params
            }
        }
        
        data_cache.set(cache_key, result)
        logger.info(f"✅ Prophet prediction completed with interval width: {interval_width:.1f}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Prophet advanced error: {e}")
        logger.info("🔄 Falling back to simple statistical prediction")
        return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))

# ============ API ENDPOINTS ============
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "NASA Paraguay Weather API",
        "version": "3.1",
        "description": "Advanced weather prediction API with ML capabilities",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API information"},
            {"path": "/api/health", "method": "GET", "description": "Health check"},
            {"path": "/api/variables", "method": "GET", "description": "Available variables"},
            {"path": "/api/forecast", "method": "GET", "description": "Weather forecast"},
            {"path": "/api/cache/stats", "method": "GET", "description": "Cache statistics"},
            {"path": "/api/cache/clear", "method": "DELETE", "description": "Clear cache"}
        ],
        "status": "operational",
        "features": ["METAR", "Open-Meteo", "Meteostat", "NASA POWER", "Prophet ML"],
        "dependencies": {
            "meteostat": METEOSTAT_AVAILABLE,
            "prophet": PROPHET_AVAILABLE
        }
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "server_date": datetime.date.today().isoformat(),
        "dependencies": {
            "meteostat": METEOSTAT_AVAILABLE,
            "prophet": PROPHET_AVAILABLE
        },
        "cache": {
            "enabled": data_cache.cache_dir is not None,
            "ttl_hours": data_cache.ttl_hours,
            "directory": data_cache.cache_dir
        },
        "version": "3.1",
        "api_name": "NASA Paraguay Weather API"
    })

@app.route("/api/variables", methods=["GET"])
def api_variables():
    """Endpoint para obtener información sobre variables disponibles"""
    variables_info = {}
    
    for var, (min_val, max_val, max_interval) in VARIABLE_LIMITS.items():
        variables_info[var] = {
            "name": var,
            "min_value": min_val,
            "max_value": max_val,
            "max_interval": max_interval,
            "unit": var.split("(")[-1].rstrip(")") if "(" in var else "",
            "description": get_variable_description(var)
        }
    
    return jsonify({
        "variables": variables_info,
        "total": len(variables_info),
        "supported_modes": ["real", "ia"],
        "data_sources": ["METAR", "Open-Meteo", "Meteostat", "NASA POWER"]
    })

def get_variable_description(variable):
    descriptions = {
        "Temperature (°C)": "Air temperature at 2 meters height",
        "Precipitation (mm)": "Total precipitation amount",
        "Wind speed (m/s)": "Wind speed at 10 meters height",
        "Humidity (%)": "Relative humidity percentage"
    }
    return descriptions.get(variable, "Weather variable")

@app.route("/api/forecast", methods=["GET"])
@rate_limit(15)
@validate_request_params
@safe_api_call("Weather forecast")
def api_forecast():
    """Endpoint principal de pronóstico meteorológico"""
    type_ = request.args.get("type", "").lower()
    if type_ not in ["real", "ia"]:
        return jsonify({"error": "Invalid type. Must be 'real' or 'ia'"}), 400
    
    variable = request.args.get("variable", "Temperature (°C)")
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
    
    logger.info(f"🔍 Forecast request: type={type_}, variable={variable}, lat={lat}, lon={lon}")
    
    if type_ == "real":
        if variable in ["Temperature (°C)", "Wind speed (m/s)", "Humidity (%)"]:
            metar = fetch_metar("SGAS")
            if metar:
                valor, unidad = None, None
                
                if "Temperature" in variable:
                    valor, unidad = metar.get("temperatura_C"), "°C"
                elif "Wind" in variable:
                    valor, unidad = metar.get("viento_mps"), "m/s"
                elif "Humidity" in variable:
                    valor, unidad = metar.get("humedad_pct"), "%"
                
                if valor is not None:
                    logger.info(f"✅ Returning METAR data: {valor} {unidad}")
                    return jsonify({
                        "fecha": metar["fecha_obs"],
                        "valor": valor,
                        "unidad": unidad,
                        "source": metar["source"],
                        "type": "observation",
                        "location": {
                            "icao": metar["icao"],
                            "description": "Asunción, Paraguay"
                        }
                    })
        
        days = request.args.get("days", default=7, type=int)
        days = max(1, min(days, 16))
        
        records, meta = fetch_hourly_openmeteo(lat, lon, variable, days)
        
        if not records:
            logger.warning("❌ No real-time data available from Open-Meteo")
            return jsonify({"error": "No real-time data available for this location"}), 404
        
        logger.info(f"✅ Returning {len(records)} hourly records from Open-Meteo")
        
        return jsonify({
            "metadata": meta,
            "data": records,
            "count": len(records),
            "granularity": "hourly"
        })
    
    elif type_ == "ia":
        date_str = request.args.get("date")
        if not date_str:
            return jsonify({"error": "Missing 'date' parameter for IA mode"}), 400
        
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        today = datetime.date.today()
        years_back = request.args.get("years", default=10, type=int)
        years_back = max(1, min(years_back, 30))
        
        if target_date < today:
            logger.info(f"📊 Historical analysis for {target_date}")
            result = get_historical_day_average(lat, lon, target_date, variable, years_back)
            
            if not result:
                return jsonify({
                    "error": f"No historical data available for {target_date}",
                    "suggestion": "Try a different date or location with more historical data"
                }), 404
            
            return jsonify({
                **result,
                "analysis_type": "historical_average",
                "granularity": "daily"
            })
        else:
            logger.info(f"🔮 Future prediction for {target_date}")
            result = train_predict_daily_advanced(lat, lon, target_date, variable, years_back)
            
            if not result:
                return jsonify({
                    "error": "Unable to generate prediction",
                    "reason": "Insufficient historical data for this location/variable combination"
                }), 500
            
            return jsonify({
                **result,
                "analysis_type": "future_prediction",
                "granularity": "daily"
            })

@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    """Estadísticas del caché"""
    try:
        cache_dir = data_cache.cache_dir
        
        if not cache_dir or not os.path.exists(cache_dir):
            return jsonify({
                "cache_enabled": False,
                "cache_directory": cache_dir,
                "status": "directory_not_found",
                "total_files": 0,
                "total_size_mb": 0
            })
        
        files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
        total_size = sum(
            os.path.getsize(os.path.join(cache_dir, f))
            for f in files if os.path.exists(os.path.join(cache_dir, f))
        )
        
        return jsonify({
            "cache_enabled": True,
            "cache_directory": cache_dir,
            "ttl_hours": data_cache.ttl_hours,
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "status": "operational"
        })
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({"error": f"Failed to get cache stats: {str(e)}"}), 500

@app.route("/api/cache/clear", methods=["DELETE"])
def clear_cache():
    """Limpiar caché completamente"""
    try:
        cache_dir = data_cache.cache_dir
        files_removed = 0
        
        if cache_dir and os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith('.pkl'):
                    try:
                        file_path = os.path.join(cache_dir, filename)
                        os.remove(file_path)
                        files_removed += 1
                    except OSError as e:
                        logger.warning(f"Could not remove cache file {filename}: {e}")
        
        logger.info(f"🧹 Cache cleared: {files_removed} files removed")
        
        return jsonify({
            "message": "Cache cleared successfully",
            "files_removed": files_removed,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "/", "/api/health", "/api/variables", 
            "/api/forecast", "/api/cache/stats", "/api/cache/clear"
        ]
    }), 404

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please wait before trying again.",
        "retry_after": 60
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"💥 Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "The server encountered an unexpected condition",
        "contact": "Check server logs for details"
    }), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad request",
        "message": "The request could not be understood by the server due to malformed syntax"
    }), 400

# ============ MAIN APPLICATION ============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    env = os.environ.get("FLASK_ENV", "development")
    debug_mode = env == "development"
    
    logger.info("🌤️  Starting NASA Paraguay Weather API v3.1")
    logger.info(f"🌍 Environment: {env}")
    logger.info(f"🔒 CORS origins: {get_allowed_origins()}")
    logger.info(f"🚀 Server starting on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"📊 Dependencies: Meteostat={METEOSTAT_AVAILABLE}, Prophet={PROPHET_AVAILABLE}")
    logger.info(f"💾 Cache: {data_cache.cache_dir} (TTL: {data_cache.ttl_hours}h)")
    logger.info("📡 Ready to serve weather predictions with ML capabilities!")
    
    cleared_on_startup = data_cache.clear_expired()
    if cleared_on_startup > 0:
        logger.info(f"🧹 Cleared {cleared_on_startup} expired cache files on startup")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
        threaded=True
    )
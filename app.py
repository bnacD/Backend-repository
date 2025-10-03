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
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WEATHERAPI_KEY = os.environ.get("WEATHERAPI_KEY", "")
VISUALCROSSING_KEY = os.environ.get("VISUALCROSSING_KEY", "")

VARIABLE_LIMITS = {
    "Temperature (°C)": (-50, 60, 30),
    "Precipitation (mm)": (0, 500, 100),
    "Wind speed (m/s)": (0, 50, 20),
    "Humidity (%)": (0, 100, 40)
}

# ============ SIMPLE RATE LIMITER ============
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

rate_limiter = SimpleRateLimiter()

# ============ DATA CACHE CLASS ============
class DataCache:
    def __init__(self, cache_dir="./cache", ttl_hours=1):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directory: {e}")
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
            logger.warning(f"Cache read error: {e}")
            return None
    def set(self, key, data):
        if not self.cache_dir:
            return
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

data_cache = DataCache()

# ============ DECORATORS ============
def rate_limit(limit=10):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_id = request.remote_addr
            if not rate_limiter.is_allowed(client_id, limit):
                return jsonify({"error": "Rate limit exceeded"}), 429
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_request_params(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            lat = float(request.args.get("lat", 0))
            lon = float(request.args.get("lon", 0))
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return jsonify({"error": "Invalid coordinates range"}), 400
        except (TypeError, ValueError):
            return jsonify({"error": "Missing or invalid lat/lon parameters"}), 400
        allowed_variables = list(VARIABLE_LIMITS.keys())
        variable = request.args.get("variable", "Temperature (°C)")
        if variable not in allowed_variables:
            return jsonify({
                "error": "Invalid variable",
                "allowed": allowed_variables
            }), 400
        return f(*args, **kwargs)
    return decorated_function

# ============ UTILITY FUNCTIONS ============
def clean_invalid_values(df):
    if df.empty:
        return df
    df = df.copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    df = df[np.isfinite(df['value'])]
    df = df[df['value'] != -999]
    df = df[df['value'] != -9999]
    return df

def remove_outliers(df, variable):
    if df.empty or len(df) < 10:
        return df
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
    df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    return df

# ============ DATA FETCHING FUNCTIONS ============
def fetch_metar(icao_code="SGAS"):
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{icao_code}.TXT"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return None
        fecha_obs = lines[0]
        metar_data = lines[1]
        result = {
            "icao": icao_code,
            "fecha_obs": fecha_obs,
            "metar_raw": metar_data,
            "source": "NOAA METAR"
        }
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
        return result
    except Exception as e:
        logger.warning(f"METAR fetch error: {e}")
        return None

def fetch_hourly_openmeteo(lat, lon, variable, days=7):
    try:
        var_mapping = {
            "Temperature (°C)": "temperature_2m",
            "Precipitation (mm)": "precipitation",
            "Wind speed (m/s)": "windspeed_10m",
            "Humidity (%)": "relativehumidity_2m"
        }
        param = var_mapping.get(variable)
        if not param:
            return pd.DataFrame(), {}
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
            return pd.DataFrame(), {}
        df = pd.DataFrame({
            "datetime": pd.to_datetime(times),
            "value": values
        })
        df = clean_invalid_values(df)
        metadata = {
            "source": "Open-Meteo",
            "location": {"latitude": lat, "longitude": lon},
            "variable": variable,
            "forecast_days": days,
            "timezone": data.get("timezone", "UTC")
        }
        return df, metadata
    except Exception as e:
        logger.error(f"Open-Meteo fetch error: {e}")
        return pd.DataFrame(), {}

def fetch_meteostat_daily(lat, lon, start, end, variable):
    if not METEOSTAT_AVAILABLE:
        return pd.DataFrame()
    column_mapping = {
        "Temperature (°C)": "tavg",
        "Precipitation (mm)": "prcp",
        "Wind speed (m/s)": "wspd",
        "Humidity (%)": "rhum"
    }
    column = column_mapping.get(variable)
    if not column:
        return pd.DataFrame()
    try:
        station_point = Point(lat, lon)
        if isinstance(start, datetime.date) and not isinstance(start, datetime.datetime):
            start_dt = datetime.datetime.combine(start, datetime.time.min)
        else:
            start_dt = start
        if isinstance(end, datetime.date) and not isinstance(end, datetime.datetime):
            end_dt = datetime.datetime.combine(end, datetime.time.max)
        else:
            end_dt = end
        logger.info(f"Fetching Meteostat data from {start_dt.date()} to {end_dt.date()}")
        df = Daily(station_point, start=start_dt, end=end_dt).fetch()
        if df.empty or column not in df.columns:
            return pd.DataFrame()
        df['value'] = df[column]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.dropna(subset=['value'], inplace=True)
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        return clean_invalid_values(df)
    except Exception as e:
        logger.error(f"Meteostat daily fetch error: {e}")
        return pd.DataFrame()

def fetch_nasa_daily(lat, lon, start, end, variable):
    var_map = {
        "Temperature (°C)": "T2M",
        "Precipitation (mm)": "PRECTOTCORR",
        "Wind speed (m/s)": "WS10M",
        "Humidity (%)": "RH2M"
    }
    param = var_map.get(variable)
    if not param:
        return pd.DataFrame()
    try:
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={param}&community=RE&longitude={lon}&latitude={lat}"
            f"&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}&format=JSON"
        )
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json().get("properties", {}).get("parameter", {}).get(param, {})
        records = []
        for date_str, value in data.items():
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                if start <= date_obj <= end and value is not None:
                    records.append({"date": date_obj, "value": float(value)})
            except (ValueError, TypeError):
                continue
        return clean_invalid_values(pd.DataFrame(records))
    except Exception as e:
        logger.error(f"NASA POWER fetch error: {e}")
        return pd.DataFrame()

# ============ PREDICTION FUNCTIONS ============
def get_historical_day_average(lat, lon, target_date, variable, years_window=10):
    cache_key = data_cache._get_cache_key(lat, lon, target_date, variable, f"hist{years_window}")
    cached_result = data_cache.get(cache_key)
    if cached_result:
        logger.info(f"Using cached historical data for {target_date}")
        return cached_result
    records = []
    for year_offset in range(years_window):
        try:
            query_year = target_date.year - year_offset - 1
            if target_date.month == 2 and target_date.day == 29:
                try:
                    query_date = datetime.date(query_year, 2, 29)
                except ValueError:
                    query_date = datetime.date(query_year, 2, 28)
            else:
                query_date = datetime.date(query_year, target_date.month, target_date.day)
            df = fetch_meteostat_daily(lat, lon, query_date, query_date, variable)
            if df.empty:
                df = fetch_nasa_daily(lat, lon, query_date, query_date, variable)
            if not df.empty:
                matching_rows = df[df['date'] == query_date]
                if not matching_rows.empty:
                    records.append({
                        "year": query_year,
                        "date": query_date.isoformat(),
                        "value": float(matching_rows['value'].iloc[0])
                    })
        except Exception as e:
            logger.warning(f"Error processing year {query_year}: {e}")
            continue
    if not records:
        return None
    values = [r['value'] for r in records]
    result = {
        "date": target_date.isoformat(),
        "average": round(float(np.mean(values)), 2),
        "min": round(float(min(values)), 2),
        "max": round(float(max(values)), 2),
        "std_dev": round(float(np.std(values)), 2) if len(values) > 1 else 0.0,
        "samples": len(records),
        "years_covered": [r['year'] for r in records],
        "years_data": records,
        "source": "Historical observations"
    }
    data_cache.set(cache_key, result)
    return result

def simple_prediction(lat, lon, target_date, variable, years_window=5):
    cache_key = data_cache._get_cache_key(lat, lon, target_date, variable, f"pred{years_window}")
    cached_result = data_cache.get(cache_key)
    if cached_result:
        logger.info(f"Using cached simple prediction for {target_date}")
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
        except Exception:
            continue
    if len(historical_values) < 2:
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
        "historical_values": [round(float(v), 2) for v in historical_values],
        "std_deviation": round(float(std_value), 2),
        "data_quality": {
            "interval_width": round(float(upper_bound - lower_bound), 2),
            "method": "statistical_average"
        }
    }
    data_cache.set(cache_key, result)
    return result

def train_predict_daily_advanced(lat, lon, target_date, variable, years_window=10):
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet not available, using statistical fallback")
        return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
    try:
        start_date = target_date - datetime.timedelta(days=365 * years_window)
        end_date = target_date - datetime.timedelta(days=1)
        cache_key = data_cache._get_cache_key(lat, lon, start_date, end_date, f"{variable}_prophet_{years_window}")
        cached_result = data_cache.get(cache_key)
        if cached_result:
            logger.info(f"Using cached Prophet prediction for {target_date}")
            return cached_result
        hist_df = fetch_meteostat_daily(lat, lon, start_date, end_date, variable)
        if hist_df.empty:
            hist_df = fetch_nasa_daily(lat, lon, start_date, end_date, variable)
        if hist_df.empty or len(hist_df) < 100:
            logger.warning(f"Insufficient data for Prophet: {len(hist_df)} records")
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        original_count = len(hist_df)
        hist_df = remove_outliers(hist_df, variable)
        outliers_removed = original_count - len(hist_df)
        if len(hist_df) < 50:
            logger.warning(f"Insufficient data after outlier removal: {len(hist_df)}")
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
            logger.warning(f"Wide prediction interval: {interval_width:.1f}")
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        result = {
            "date": target_date.isoformat(),
            "predicted_value": round(predicted_value, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "confidence": "80%",
            "model": "Prophet (Advanced)",
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
        return result
    except Exception as e:
        logger.error(f"Prophet Advanced error: {e}")
        return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))

# ============ API ENDPOINTS ============
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "name": "Weather Prediction API",
        "version": "3.0",
        "endpoints": [
            "/api/health",
            "/api/forecast",
            "/api/cache/stats",
            "/api/cache/clear"
        ]
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
            "enabled": True,
            "ttl_hours": data_cache.ttl_hours,
            "directory": data_cache.cache_dir
        },
        "version": "3.0"
    })

@app.route("/api/forecast", methods=["GET"])
@rate_limit(10)
@validate_request_params
def api_forecast():
    type_ = request.args.get("type", "").lower()
    if type_ not in ["real", "ia"]:
        return jsonify({"error": "Invalid type. Must be 'real' or 'ia'"}), 400
    variable = request.args.get("variable", "Temperature (°C)")
    lat = float(request.args.get("lat"))
    lon = float(request.args.get("lon"))
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
                    return jsonify({
                        "fecha": metar["fecha_obs"],
                        "valor": valor,
                        "unidad": unidad,
                        "source": metar["source"],
                        "type": "observation"
                    })
        days = request.args.get("days", default=7, type=int)
        days = max(1, min(days, 16))
        df, meta = fetch_hourly_openmeteo(lat, lon, variable, days)
        if df.empty:
            return jsonify({"error": "No real-time data available"}), 404
        return jsonify({
            "metadata": meta,
            "data": df.to_dict(orient="records"),
            "count": len(df),
            "granularity": "hourly"
        })
    elif type_ == "ia":
        date_str = request.args.get("date")
        if not date_str:
            return jsonify({"error": "Missing 'date' parameter"}), 400
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        today = datetime.date.today()
        years_back = request.args.get("years", default=10, type=int)
        years_back = max(1, min(years_back, 30))
        if target_date < today:
            result = get_historical_day_average(lat, lon, target_date, variable, years_back)
            if not result:
                return jsonify({"error": f"No historical data available for {target_date}"}), 404
            return jsonify({
                **result,
                "analysis_type": "historical_average",
                "granularity": "daily"
            })
        else:
            result = train_predict_daily_advanced(lat, lon, target_date, variable, years_back)
            if not result:
                return jsonify({"error": "Unable to generate prediction"}), 500
            return jsonify({
                **result,
                "analysis_type": "future_prediction",
                "granularity": "daily"
            })

@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    try:
        cache_dir = data_cache.cache_dir
        if not os.path.exists(cache_dir):
            return jsonify({
                "cache_enabled": True,
                "cache_directory": cache_dir,
                "status": "directory_not_found",
                "total_files": 0,
                "total_size_mb": 0
            })
        files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
        total_size = sum(
            os.path.getsize(os.path.join(cache_dir, f))
            for f in files
        )
        return jsonify({
            "cache_enabled": True,
            "cache_directory": cache_dir,
            "ttl_hours": data_cache.ttl_hours,
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get cache stats: {str(e)}"}), 500

@app.route("/api/cache/clear", methods=["DELETE"])
def clear_cache():
    try:
        cache_dir = data_cache.cache_dir
        files_removed = 0
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith('.pkl'):
                    try:
                        os.remove(os.path.join(cache_dir, filename))
                        files_removed += 1
                    except OSError:
                        pass
        return jsonify({
            "message": "Cache cleared successfully",
            "files_removed": files_removed
        })
    except Exception as e:
        return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/",
            "/api/health",
            "/api/forecast",
            "/api/cache/stats",
            "/api/cache/clear"
        ]
    }), 404

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests"
    }), 429

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "Please check server logs for details"
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
    logger.info("🌤️ Starting Weather API v3.0")
    logger.info(f"🚀 Server starting on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    logger.info(f"📊 Dependencies: Meteostat={METEOSTAT_AVAILABLE}, Prophet={PROPHET_AVAILABLE}")
    logger.info(f"💾 Cache: {data_cache.cache_dir} (TTL: {data_cache.ttl_hours}h)")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode,
        threaded=True
    )
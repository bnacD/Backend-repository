from flask import Flask, request, jsonify
from flask_cors import CORS
from meteostat import Point, Daily
from prophet import Prophet
import datetime
import pandas as pd
import numpy as np
import requests
import re
import math
import os
from typing import Tuple, Dict, Any, Optional

app = Flask(__name__)

# ============ CORS CONFIGURATION ============
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# ============ API KEYS ============
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
VISUALCROSSING_KEY = os.getenv("VISUALCROSSING_KEY", "")

if not WEATHERAPI_KEY or not VISUALCROSSING_KEY:
    print("⚠️  WARNING: API keys not set. Some features may not work.")

# ============ VARIABLE LIMITS ============
VARIABLE_LIMITS = {
    'Temperature (°C)': (-50, 60, 20),  # (min, max, max_interval)
    'Precipitation (mm)': (0, 500, 100),
    'Wind speed (m/s)': (0, 50, 15),
    'Humidity (%)': (0, 100, 30)
}

# ============ UTILITIES ============
def validate_coordinates(lat: float, lon: float) -> Tuple[bool, str]:
    if not (-90 <= lat <= 90):
        return False, "Latitude must be between -90 and 90"
    if not (-180 <= lon <= 180):
        return False, "Longitude must be between -180 and 180"
    return True, ""

def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    invalid_values = [999, -9999, -999]
    df = df[~df['value'].isin(invalid_values)]
    df = df[df['value'] >= 0]
    return df

def smooth_series(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    if not df.empty and 'value' in df.columns:
        df['value'] = df['value'].rolling(window=window, min_periods=1).mean()
    return df

def remove_outliers(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Remove statistical outliers using IQR method"""
    if df.empty or 'value' not in df.columns:
        return df
    
    # Get variable limits
    min_limit, max_limit, _ = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
    
    # First, apply physical limits
    df = df[(df['value'] >= min_limit) & (df['value'] <= max_limit)]
    
    if len(df) < 10:  # Need minimum data for IQR
        return df
    
    # Apply IQR method
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR > 0:  # Avoid division by zero
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    
    return df

# ============ METAR PARSING ============
def fetch_metar(station_code: str = "SGAS", max_age_minutes: int = 90) -> Dict[str, Any]:
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station_code}.TXT"
        r = requests.get(url, timeout=5)
        r.raise_for_status()

        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return {}
            
        fecha_obs_str = lines[0].strip()
        metar_line = lines[1].strip()

        fecha_obs = datetime.datetime.strptime(fecha_obs_str, "%Y/%m/%d %H:%M")
        age_seconds = (datetime.datetime.utcnow() - fecha_obs).total_seconds()
        if age_seconds > (max_age_minutes * 60):
            return {}

        temp_match = re.search(r"(\d{2})/(\d{2})", metar_line)
        temperatura_C = int(temp_match.group(1)) if temp_match else None
        dewpoint_C = int(temp_match.group(2)) if temp_match else None

        humedad_pct = None
        if temperatura_C is not None and dewpoint_C is not None:
            es = 6.11 * math.exp(17.62 * temperatura_C / (243.12 + temperatura_C))
            e = 6.11 * math.exp(17.62 * dewpoint_C / (243.12 + dewpoint_C))
            humedad_pct = round((e / es) * 100)

        viento_match = re.search(r"(\d{3})(\d{2})KT", metar_line)
        viento_mps = None
        if viento_match:
            velocidad_nudos = int(viento_match.group(2))
            viento_mps = round(velocidad_nudos * 0.514444, 1)

        presion_match = re.search(r"Q(\d{4})", metar_line)
        presion_hPa = int(presion_match.group(1)) if presion_match else None

        return {
            "source": f"DINAC / METAR {station_code}",
            "fecha_obs": fecha_obs_str,
            "temperatura_C": temperatura_C,
            "humedad_pct": humedad_pct,
            "viento_mps": viento_mps,
            "presion_hPa": presion_hPa
        }
    except Exception as e:
        print(f"❌ [ERROR METAR] {e}")
        return {}

# ============ HOURLY APIS ============
def fetch_hourly_openmeteo(lat: float, lon: float, variable: str, days_ahead: int = 7) -> Tuple[pd.DataFrame, Dict]:
    param_map = {
        "Temperature (°C)": "temperature_2m",
        "Precipitation (mm)": "precipitation",
        "Wind speed (m/s)": "windspeed_10m",
        "Humidity (%)": "relativehumidity_2m"
    }
    parameter = param_map.get(variable)
    if not parameter:
        return pd.DataFrame(), {}
        
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={parameter}&timezone=auto&forecast_days={days_ahead}"
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        data = r.json().get("hourly", {})
        if not data or parameter not in data:
            return pd.DataFrame(), {}
            
        df = pd.DataFrame({
            "date": pd.to_datetime(data["time"]), 
            "value": data[parameter]
        })
        
        return smooth_series(df), {
            "units": "metric", 
            "source": "Open-Meteo",
            "type": "modeled"
        }
    except Exception as e:
        print(f"❌ [ERROR Open-Meteo] {e}")
        return pd.DataFrame(), {}

# ============ DAILY HISTORICAL DATA ============
def fetch_meteostat_daily(lat: float, lon: float, start: datetime.date, 
                          end: datetime.date, variable: str) -> pd.DataFrame:
    """
    ✅ CORREGIDO: Manejo correcto de tipos datetime
    """
    col_map = {
        "Temperature (°C)": "tavg",
        "Precipitation (mm)": "prcp",
        "Wind speed (m/s)": "wspd",
        "Humidity (%)": "rhum"
    }
    
    column = col_map.get(variable)
    if not column:
        return pd.DataFrame()
        
    try:
        station_point = Point(lat, lon)
        
        # ✅ FIX: Convertir date a datetime correctamente
        if isinstance(start, datetime.date) and not isinstance(start, datetime.datetime):
            start_dt = datetime.datetime.combine(start, datetime.time.min)
        else:
            start_dt = start
            
        if isinstance(end, datetime.date) and not isinstance(end, datetime.datetime):
            end_dt = datetime.datetime.combine(end, datetime.time.max)
        else:
            end_dt = end
        
        print(f"📊 Fetching daily historical data from {start_dt.date()} to {end_dt.date()}")
        
        df = Daily(station_point, start=start_dt, end=end_dt).fetch()
        
        if df.empty:
            return pd.DataFrame()
            
        df['value'] = df[column]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.dropna(subset=['value'], inplace=True)
        
        # ✅ Validar rango de fechas
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        
        return clean_invalid_values(df)
        
    except Exception as e:
        print(f"❌ [ERROR Meteostat daily] {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def fetch_nasa_daily(lat: float, lon: float, start: datetime.date, 
                     end: datetime.date, variable: str) -> pd.DataFrame:
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
        
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        data = r.json().get("properties", {}).get("parameter", {}).get(param, {})
        records = []
        
        for date_str, value in data.items():
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                if start <= date_obj <= end:
                    records.append({"date": date_obj, "value": value})
            except ValueError:
                continue
        
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"❌ [ERROR NASA POWER daily] {e}")
        return pd.DataFrame()

# ============ IMPROVED AI FUNCTIONS ============
def get_historical_day_average(lat: float, lon: float, target_date: datetime.date, 
                                variable: str, years_window: int = 10) -> Optional[Dict[str, Any]]:
    records = []
    
    for year_offset in range(years_window):
        try:
            query_year = target_date.year - year_offset
            
            # Handle leap year
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
            print(f"⚠️  Error processing year {query_year}: {e}")
            continue
    
    if not records:
        return None
    
    values = [r['value'] for r in records]
    
    return {
        "date": target_date.isoformat(),
        "average": round(float(pd.Series(values).mean()), 2),
        "min": round(float(min(values)), 2),
        "max": round(float(max(values)), 2),
        "std_dev": round(float(pd.Series(values).std()), 2) if len(values) > 1 else 0.0,
        "samples": len(records),
        "years_covered": [r['year'] for r in records],
        "years_data": records,
        "source": "Historical observations"
    }

def simple_prediction(lat: float, lon: float, target_date: datetime.date, 
                     variable: str, years_window: int = 5) -> Optional[Dict[str, Any]]:
    """
    Predicción simple basada en promedios históricos del mismo día en años anteriores
    Más confiable que Prophet para datos limitados
    """
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
            
            # Buscar datos en ventana de ±3 días para mayor robustez
            start_window = query_date - datetime.timedelta(days=3)
            end_window = query_date + datetime.timedelta(days=3)
            
            df = fetch_meteostat_daily(lat, lon, start_window, end_window, variable)
            if df.empty:
                df = fetch_nasa_daily(lat, lon, start_window, end_window, variable)
            
            if not df.empty:
                # Remove outliers
                df = remove_outliers(df, variable)
                if not df.empty:
                    avg_value = df['value'].mean()
                    if not pd.isna(avg_value):
                        historical_values.append(avg_value)
                        
        except Exception as e:
            continue
    
    if len(historical_values) < 2:
        return None
    
    # Calcular estadísticas
    mean_value = np.mean(historical_values)
    std_value = np.std(historical_values)
    
    # Aplicar límites físicos
    min_limit, max_limit, max_interval = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
    
    predicted_value = max(min_limit, min(max_limit, mean_value))
    lower_bound = max(min_limit, min(max_limit, mean_value - std_value))
    upper_bound = max(min_limit, min(max_limit, mean_value + std_value))
    
    return {
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

def train_predict_daily_improved(lat: float, lon: float, target_date: datetime.date, 
                                variable: str, years_window: int = 10) -> Optional[Dict[str, Any]]:
    """
    Predicción mejorada con Prophet optimizado
    """
    try:
        # Obtener datos históricos
        start_date = target_date - datetime.timedelta(days=365 * years_window)
        end_date = target_date - datetime.timedelta(days=1)
        
        hist_df = fetch_meteostat_daily(lat, lon, start_date, end_date, variable)
        if hist_df.empty:
            hist_df = fetch_nasa_daily(lat, lon, start_date, end_date, variable)
        
        if hist_df.empty or len(hist_df) < 100:
            print(f"⚠️  Insufficient data for Prophet: {len(hist_df)} records")
            # Fallback a predicción simple
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        
        # Limpiar outliers
        original_count = len(hist_df)
        hist_df = remove_outliers(hist_df, variable)
        outliers_removed = original_count - len(hist_df)
        
        if## 📁 **Códigos Completos con Todas las Mejoras Implementadas**

### **1. app.py** (Backend Optimizado Completo)

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from meteostat import Point, Daily
from prophet import Prophet
import datetime
import pandas as pd
import numpy as np
import requests
import re
import math
import os
from typing import Tuple, Dict, Any, Optional

app = Flask(__name__)

# ============ CORS CONFIGURATION ============
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# ============ API KEYS ============
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
VISUALCROSSING_KEY = os.getenv("VISUALCROSSING_KEY", "")

if not WEATHERAPI_KEY or not VISUALCROSSING_KEY:
    print("⚠️  WARNING: API keys not set. Some features may not work.")

# ============ VARIABLE LIMITS ============
VARIABLE_LIMITS = {
    'Temperature (°C)': (-50, 60, 20),  # (min, max, max_interval)
    'Precipitation (mm)': (0, 500, 100),
    'Wind speed (m/s)': (0, 50, 15),
    'Humidity (%)': (0, 100, 30)
}

# ============ UTILITIES ============
def validate_coordinates(lat: float, lon: float) -> Tuple[bool, str]:
    if not (-90 <= lat <= 90):
        return False, "Latitude must be between -90 and 90"
    if not (-180 <= lon <= 180):
        return False, "Longitude must be between -180 and 180"
    return True, ""

def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    invalid_values = [999, -9999, -999]
    df = df[~df['value'].isin(invalid_values)]
    df = df[df['value'] >= 0]
    return df

def smooth_series(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    if not df.empty and 'value' in df.columns:
        df['value'] = df['value'].rolling(window=window, min_periods=1).mean()
    return df

def remove_outliers(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Remove statistical outliers using IQR method"""
    if df.empty or 'value' not in df.columns:
        return df
    
    # Get variable limits
    min_limit, max_limit, _ = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
    
    # First, apply physical limits
    df = df[(df['value'] >= min_limit) & (df['value'] <= max_limit)]
    
    if len(df) < 10:  # Need minimum data for IQR
        return df
    
    # Apply IQR method
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR > 0:  # Avoid division by zero
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    
    return df

# ============ METAR PARSING ============
def fetch_metar(station_code: str = "SGAS", max_age_minutes: int = 90) -> Dict[str, Any]:
    try:
        url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{station_code}.TXT"
        r = requests.get(url, timeout=5)
        r.raise_for_status()

        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return {}
            
        fecha_obs_str = lines[0].strip()
        metar_line = lines[1].strip()

        fecha_obs = datetime.datetime.strptime(fecha_obs_str, "%Y/%m/%d %H:%M")
        age_seconds = (datetime.datetime.utcnow() - fecha_obs).total_seconds()
        if age_seconds > (max_age_minutes * 60):
            return {}

        temp_match = re.search(r"(\d{2})/(\d{2})", metar_line)
        temperatura_C = int(temp_match.group(1)) if temp_match else None
        dewpoint_C = int(temp_match.group(2)) if temp_match else None

        humedad_pct = None
        if temperatura_C is not None and dewpoint_C is not None:
            es = 6.11 * math.exp(17.62 * temperatura_C / (243.12 + temperatura_C))
            e = 6.11 * math.exp(17.62 * dewpoint_C / (243.12 + dewpoint_C))
            humedad_pct = round((e / es) * 100)

        viento_match = re.search(r"(\d{3})(\d{2})KT", metar_line)
        viento_mps = None
        if viento_match:
            velocidad_nudos = int(viento_match.group(2))
            viento_mps = round(velocidad_nudos * 0.514444, 1)

        presion_match = re.search(r"Q(\d{4})", metar_line)
        presion_hPa = int(presion_match.group(1)) if presion_match else None

        return {
            "source": f"DINAC / METAR {station_code}",
            "fecha_obs": fecha_obs_str,
            "temperatura_C": temperatura_C,
            "humedad_pct": humedad_pct,
            "viento_mps": viento_mps,
            "presion_hPa": presion_hPa
        }
    except Exception as e:
        print(f"❌ [ERROR METAR] {e}")
        return {}

# ============ HOURLY APIS ============
def fetch_hourly_openmeteo(lat: float, lon: float, variable: str, days_ahead: int = 7) -> Tuple[pd.DataFrame, Dict]:
    param_map = {
        "Temperature (°C)": "temperature_2m",
        "Precipitation (mm)": "precipitation",
        "Wind speed (m/s)": "windspeed_10m",
        "Humidity (%)": "relativehumidity_2m"
    }
    parameter = param_map.get(variable)
    if not parameter:
        return pd.DataFrame(), {}
        
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={parameter}&timezone=auto&forecast_days={days_ahead}"
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        data = r.json().get("hourly", {})
        if not data or parameter not in data:
            return pd.DataFrame(), {}
            
        df = pd.DataFrame({
            "date": pd.to_datetime(data["time"]), 
            "value": data[parameter]
        })
        
        return smooth_series(df), {
            "units": "metric", 
            "source": "Open-Meteo",
            "type": "modeled"
        }
    except Exception as e:
        print(f"❌ [ERROR Open-Meteo] {e}")
        return pd.DataFrame(), {}

# ============ DAILY HISTORICAL DATA ============
def fetch_meteostat_daily(lat: float, lon: float, start: datetime.date, 
                          end: datetime.date, variable: str) -> pd.DataFrame:
    col_map = {
        "Temperature (°C)": "tavg",
        "Precipitation (mm)": "prcp",
        "Wind speed (m/s)": "wspd",
        "Humidity (%)": "rhum"
    }
    
    column = col_map.get(variable)
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
        
        print(f"📊 Fetching daily historical data from {start_dt.date()} to {end_dt.date()}")
        
        df = Daily(station_point, start=start_dt, end=end_dt).fetch()
        
        if df.empty:
            return pd.DataFrame()
            
        df['value'] = df[column]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.date
        df.dropna(subset=['value'], inplace=True)
        
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        
        return clean_invalid_values(df)
        
    except Exception as e:
        print(f"❌ [ERROR Meteostat daily] {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def fetch_nasa_daily(lat: float, lon: float, start: datetime.date, 
                     end: datetime.date, variable: str) -> pd.DataFrame:
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
        
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        data = r.json().get("properties", {}).get("parameter", {}).get(param, {})
        records = []
        
        for date_str, value in data.items():
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                if start <= date_obj <= end:
                    records.append({"date": date_obj, "value": value})
            except ValueError:
                continue
        
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"❌ [ERROR NASA POWER daily] {e}")
        return pd.DataFrame()

# ============ IMPROVED AI FUNCTIONS ============
def get_historical_day_average(lat: float, lon: float, target_date: datetime.date, 
                                variable: str, years_window: int = 10) -> Optional[Dict[str, Any]]:
    records = []
    
    for year_offset in range(years_window):
        try:
            query_year = target_date.year - year_offset
            
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
            print(f"⚠️  Error processing year {query_year}: {e}")
            continue
    
    if not records:
        return None
    
    values = [r['value'] for r in records]
    
    return {
        "date": target_date.isoformat(),
        "average": round(float(pd.Series(values).mean()), 2),
        "min": round(float(min(values)), 2),
        "max": round(float(max(values)), 2),
        "std_dev": round(float(pd.Series(values).std()), 2) if len(values) > 1 else 0.0,
        "samples": len(records),
        "years_covered": [r['year'] for r in records],
        "years_data": records,
        "source": "Historical observations"
    }

def simple_prediction(lat: float, lon: float, target_date: datetime.date, 
                     variable: str, years_window: int = 5) -> Optional[Dict[str, Any]]:
    """
    Predicción simple basada en promedios históricos del mismo día en años anteriores
    Más confiable que Prophet para datos limitados
    """
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
            
            # Buscar datos en ventana de ±3 días para mayor robustez
            start_window = query_date - datetime.timedelta(days=3)
            end_window = query_date + datetime.timedelta(days=3)
            
            df = fetch_meteostat_daily(lat, lon, start_window, end_window, variable)
            if df.empty:
                df = fetch_nasa_daily(lat, lon, start_window, end_window, variable)
            
            if not df.empty:
                # Remove outliers
                df = remove_outliers(df, variable)
                if not df.empty:
                    avg_value = df['value'].mean()
                    if not pd.isna(avg_value):
                        historical_values.append(avg_value)
                        
        except Exception as e:
            continue
    
    if len(historical_values) < 2:
        return None
    
    # Calcular estadísticas
    mean_value = np.mean(historical_values)
    std_value = np.std(historical_values)
    
    # Aplicar límites físicos
    min_limit, max_limit, max_interval = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
    
    predicted_value = max(min_limit, min(max_limit, mean_value))
    lower_bound = max(min_limit, min(max_limit, mean_value - std_value))
    upper_bound = max(min_limit, min(max_limit, mean_value + std_value))
    
    return {
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

def train_predict_daily_improved(lat: float, lon: float, target_date: datetime.date, 
                                variable: str, years_window: int = 10) -> Optional[Dict[str, Any]]:
    """
    Predicción mejorada con Prophet optimizado
    """
    try:
        # Obtener datos históricos
        start_date = target_date - datetime.timedelta(days=365 * years_window)
        end_date = target_date - datetime.timedelta(days=1)
        
        hist_df = fetch_meteostat_daily(lat, lon, start_date, end_date, variable)
        if hist_df.empty:
            hist_df = fetch_nasa_daily(lat, lon, start_date, end_date, variable)
        
        if hist_df.empty or len(hist_df) < 100:
            print(f"⚠️  Insufficient data for Prophet: {len(hist_df)} records")
            # Fallback a predicción simple
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        
        # Limpiar outliers
        original_count = len(hist_df)
        hist_df = remove_outliers(hist_df, variable)
        outliers_removed = original_count - len(hist_df)
        
        if len(hist_df) < 50:
            print(f"⚠️  Insufficient data after outlier removal: {len(hist_df)}")
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        
        hist_df = hist_df.copy()
        hist_df['ds'] = pd.to_datetime(hist_df['date'])
        hist_df['y'] = hist_df['value']
        hist_df = hist_df[['ds', 'y']].sort_values('ds')
        hist_df = hist_df.drop_duplicates(subset=['ds'])
        
        # Prophet optimizado para datos meteorológicos
        model = Prophet(
            daily_seasonality=False,
            yearly_seasonality=True,
            weekly_seasonality=False,
            changepoint_prior_scale=0.01,  # Más conservador
            seasonality_prior_scale=1.0,   # Menos agresivo
            interval_width=0.80,           # Intervalos más estrechos
            seasonality_mode='additive'    # Más estable
        )
        
        # Agregar estacionalidad mensual
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(hist_df)
        
        future = pd.DataFrame({'ds': [pd.to_datetime(target_date)]})
        forecast = model.predict(future)
        
        predicted_value = float(forecast['yhat'].iloc[0])
        lower_bound = float(forecast['yhat_lower'].iloc[0])
        upper_bound = float(forecast['yhat_upper'].iloc[0])
        
        # Aplicar límites realistas según variable
        min_limit, max_limit, max_interval = VARIABLE_LIMITS.get(variable, (-1000, 1000, 100))
        
        predicted_value = max(min_limit, min(max_limit, predicted_value))
        lower_bound = max(min_limit, min(max_limit, lower_bound))
        upper_bound = max(min_limit, min(max_limit, upper_bound))
        
        # Validar que el intervalo no sea demasiado amplio
        interval_width = upper_bound - lower_bound
        data_std = hist_df['y'].std()
        
        if interval_width > max_interval or interval_width > 6 * data_std:
            print(f"⚠️  Interval too wide ({interval_width:.1f}), using statistical fallback")
            # Fallback a método estadístico
            return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))
        
        return {
            "date": target_date.isoformat(),
            "predicted_value": round(predicted_value, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "confidence": "80%",
            "model": "Prophet (Optimized)",
            "training_samples": len(hist_df),
            "training_range": {
                "start": hist_df['ds'].min().date().isoformat(),
                "end": hist_df['ds'].max().date().isoformat()
            },
            "data_quality": {
                "outliers_removed": outliers_removed,
                "data_std": round(float(data_std), 2),
                "interval_width": round(interval_width, 2),
                "method": "prophet_ml"
            }
        }
        
    except Exception as e:
        print(f"❌ [ERROR Prophet Improved] {e}")
        # Fallback a predicción simple
        return simple_prediction(lat, lon, target_date, variable, min(years_window, 5))

# ============ API ENDPOINTS ============
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "server_date": datetime.date.today().isoformat(),
        "apis_configured": {
            "weatherapi": bool(WEATHERAPI_KEY),
            "visualcrossing": bool(VISUALCROSSING_KEY)
        },
        "version": "2.0-improved"
    })

@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    type_ = request.args.get("type", "").lower()
    if type_ not in ["real", "ia"]:
        return jsonify({"error": "Invalid type. Must be 'real' or 'ia'"}), 400
    
    variable = request.args.get("variable", "Temperature (°C)")
    
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing lat/lon"}), 400
    
    is_valid, error_msg = validate_coordinates(lat, lon)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    # ============ REAL MODE ============
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
    
    # ============ IA MODE ============
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
        
        # ============ PAST DATE ============
        if target_date < today:
            result = get_historical_day_average(lat, lon, target_date, variable, years_back)
            
            if not result:
                return jsonify({"error": f"No historical data for {target_date}"}), 404
            
            return jsonify({
                **result,
                "analysis_type": "historical_average",
                "granularity": "daily"
            })
        
        # ============ FUTURE DATE ============
        else:
            # Usar predicción mejorada
            result = train_predict_daily_improved(lat, lon, target_date, variable, years_back)
            
            if not result:
                return jsonify({"error": "Unable to generate prediction"}), 500
            
            return jsonify({
                **result,
                "analysis_type": "future_prediction",
                "granularity": "daily"
            })

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============ MAIN ============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
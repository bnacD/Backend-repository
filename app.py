from flask import Flask, request, jsonify
from flask_cors import CORS
from meteostat import Point, Daily
from prophet import Prophet
import datetime
import pandas as pd
import requests
import re
import math
import os
from typing import Tuple, Dict, Any, Optional

app = Flask(__name__)
CORS(app)

# ============ CONFIGURATION ============
WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")
VISUALCROSSING_KEY = os.getenv("VISUALCROSSING_KEY", "")

# ============ UTILITIES ============
def validate_coordinates(lat: float, lon: float) -> Tuple[bool, str]:
    """Validate latitude and longitude ranges."""
    if not (-90 <= lat <= 90):
        return False, "Latitude must be between -90 and 90"
    if not (-180 <= lon <= 180):
        return False, "Longitude must be between -180 and 180"
    return True, ""

def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid/sentinel values from dataframe."""
    invalid_values = [999, -9999, -999]
    df = df[~df['value'].isin(invalid_values)]
    df = df[df['value'] >= 0]
    return df

def smooth_series(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Apply rolling average to smooth modeled data."""
    if not df.empty and 'value' in df.columns:
        df['value'] = df['value'].rolling(window=window, min_periods=1).mean()
    return df

# ============ METAR PARSING (Sin cambios) ============
def fetch_metar(station_code: str = "SGAS", max_age_minutes: int = 90) -> Dict[str, Any]:
    """Fetch and parse METAR data from NOAA."""
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
            "source": f"DINAC / METAR {station_code} (NOAA)",
            "fecha_obs": fecha_obs_str,
            "temperatura_C": temperatura_C,
            "humedad_pct": humedad_pct,
            "viento_mps": viento_mps,
            "presion_hPa": presion_hPa,
            "raw_metar": metar_line
        }
    except Exception as e:
        print(f"❌ [ERROR METAR {station_code}]", e)
        return {}

# ============ HOURLY FORECAST APIs (Para modo REAL) ============
def fetch_hourly_nasa(lat: float, lon: float, variable: str, days_ahead: int = 7) -> Tuple[pd.DataFrame, Dict]:
    """Fetch hourly forecast from NASA POWER API."""
    var_map = {
        "Temperature (°C)": "T2M",
        "Precipitation (mm)": "PRECTOTCORR",
        "Wind speed (m/s)": "WS10M",
        "Humidity (%)": "RH2M"
    }
    param = var_map.get(variable)
    if not param:
        return pd.DataFrame(), {}
        
    end_date = datetime.date.today() + datetime.timedelta(days=days_ahead - 1)
    start_date = datetime.date.today()
    
    try:
        url = (
            f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
            f"parameters={param}&community=RE&longitude={lon}&latitude={lat}"
            f"&start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}&format=JSON"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        data = r.json().get("properties", {}).get("parameter", {}).get(param, {})
        df = pd.DataFrame([
            {"date": pd.to_datetime(ts), "value": val} 
            for ts, val in data.items()
        ])
        
        return smooth_series(df), {
            "units": "metric", 
            "source": "NASA POWER",
            "type": "modeled"
        }
    except Exception as e:
        print(f"❌ [ERROR NASA POWER hourly] {e}")
        return pd.DataFrame(), {}

def fetch_hourly_openmeteo(lat: float, lon: float, variable: str, days_ahead: int = 7) -> Tuple[pd.DataFrame, Dict]:
    """Fetch hourly forecast from Open-Meteo API."""
    param_map = {
        "Temperature (°C)": "temperature_2m",
        "Precipitation (mm)": "precipitation",
        "Wind speed (m/s)": "windspeed_10m",
        "Humidity (%)": "relativehumidity_2m"
    }
    parameter = param_map.get(variable)
    if not parameter:
        return pd.DataFrame(), {}
        
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly={parameter}"
        f"&timezone=auto&forecast_days={days_ahead}"
    )
    
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

def fetch_hourly_weatherapi(lat: float, lon: float, variable: str, days_ahead: int = 7) -> Tuple[pd.DataFrame, Dict]:
    """Fetch hourly forecast from WeatherAPI."""
    if not WEATHERAPI_KEY:
        return pd.DataFrame(), {}
        
    var_map_api = {
        "Temperature (°C)": ("temp_c", "°C"),
        "Precipitation (mm)": ("precip_mm", "mm"),
        "Wind speed (m/s)": ("wind_kph", "km/h"),
        "Humidity (%)": ("humidity", "%")
    }
    
    mapping = var_map_api.get(variable)
    if not mapping:
        return pd.DataFrame(), {}
        
    code, unit = mapping
    
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days={days_ahead}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        
        forecast = r.json().get("forecast", {}).get("forecastday", [])
        records = []
        
        for day in forecast:
            for hour in day.get("hour", []):
                value = hour.get(code)
                if code == "wind_kph":
                    value = value / 3.6
                    unit = "m/s"
                    
                records.append({
                    "date": pd.to_datetime(hour["time"]), 
                    "value": value
                })
                
        df = pd.DataFrame(records)
        return smooth_series(df), {
            "units": unit, 
            "source": "WeatherAPI",
            "type": "mixed"
        }
    except Exception as e:
        print(f"❌ [ERROR WeatherAPI] {e}")
        return pd.DataFrame(), {}

# ============ DAILY HISTORICAL DATA (Para modo IA) ============
def fetch_meteostat_daily(lat: float, lon: float, start: datetime.date, 
                          end: datetime.date, variable: str) -> pd.DataFrame:
    """
    Fetch DAILY aggregated historical data from Meteostat.
    Returns one value per day (not hourly).
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
        df = Daily(station_point, start=start, end=end).fetch()
        
        if df.empty:
            return pd.DataFrame()
            
        df['value'] = df[column]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.date  # Ensure date type
        df.dropna(subset=['value'], inplace=True)
        
        return clean_invalid_values(df)
    except Exception as e:
        print(f"❌ [ERROR Meteostat daily] {e}")
        return pd.DataFrame()

def fetch_visualcrossing_daily(lat: float, lon: float, start: datetime.date, 
                               end: datetime.date, variable: str) -> pd.DataFrame:
    """Fetch DAILY historical data from Visual Crossing."""
    if not VISUALCROSSING_KEY:
        return pd.DataFrame()
        
    col_map = {
        "Temperature (°C)": "temp",
        "Precipitation (mm)": "precip",
        "Wind speed (m/s)": "windspeed",
        "Humidity (%)": "humidity"
    }
    
    column = col_map.get(variable)
    if not column:
        return pd.DataFrame()
        
    try:
        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
            f"{lat},{lon}/{start}/{end}"
        )
        params = {
            "unitGroup": "metric", 
            "key": VISUALCROSSING_KEY, 
            "include": "days"
        }
        
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        
        data = r.json().get("days", [])
        df = pd.DataFrame([
            {"date": datetime.datetime.strptime(d["datetime"], "%Y-%m-%d").date(), 
             "value": d.get(column)} 
            for d in data
        ])
        df.dropna(subset=['value'], inplace=True)
        
        return df
    except Exception as e:
        print(f"❌ [ERROR VisualCrossing daily] {e}")
        return pd.DataFrame()

def fetch_nasa_daily(lat: float, lon: float, start: datetime.date, 
                     end: datetime.date, variable: str) -> pd.DataFrame:
    """Fetch DAILY historical data from NASA POWER."""
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
        df = pd.DataFrame([
            {"date": datetime.datetime.strptime(k, "%Y%m%d").date(), "value": v} 
            for k, v in data.items()
        ])
        
        return df
    except Exception as e:
        print(f"❌ [ERROR NASA POWER daily] {e}")
        return pd.DataFrame()

# ============ AI PREDICTION ============
def train_predict_model(hist_df: pd.DataFrame, predict_date: datetime.date) -> Tuple[Optional[float], pd.DataFrame]:
    """
    Train Prophet model and generate prediction for a specific date.
    
    Returns:
        Tuple of (predicted_value, forecast_df)
    """
    try:
        if hist_df.empty or len(hist_df) < 10:
            print("⚠️  Insufficient historical data for training")
            return None, pd.DataFrame()
            
        # Prepare data for Prophet
        hist_df = hist_df.copy()
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        hist_df = hist_df.rename(columns={'date': 'ds', 'value': 'y'})
        
        # Configure and train Prophet
        model = Prophet(
            daily_seasonality=False,  # Not needed for daily data
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )
        
        model.fit(hist_df)
        
        # Generate prediction for target date
        future = pd.DataFrame({'ds': [pd.to_datetime(predict_date)]})
        forecast = model.predict(future)
        
        predicted_value = forecast['yhat'].iloc[0]
        
        return float(predicted_value), forecast
    except Exception as e:
        print(f"❌ [ERROR Prophet] {e}")
        return None, pd.DataFrame()

def get_historical_day_value(hist_df: pd.DataFrame, target_date: datetime.date) -> Optional[float]:
    """
    Retrieve actual historical value for a specific past date.
    
    Returns:
        The observed value for that day, or None if not available
    """
    try:
        hist_df = hist_df.copy()
        hist_df['date'] = pd.to_datetime(hist_df['date']).dt.date
        
        day_data = hist_df[hist_df['date'] == target_date]
        
        if not day_data.empty:
            return float(day_data['value'].iloc[0])
        else:
            return None
    except Exception as e:
        print(f"❌ [ERROR retrieving historical day] {e}")
        return None

# ============ API ENDPOINTS ============
@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    """
    Main forecast endpoint supporting real-time and AI prediction modes.
    
    Query Parameters:
        - type: "real" or "ia"
        - variable: Weather variable to query
        - lat: Latitude
        - lon: Longitude
        
    For type="real":
        - days: Forecast days (default: 7)
        
    For type="ia":
        - date: Target date in YYYY-MM-DD format
        - years: Historical years to use for training (default: 10)
    """
    # ============ PARAMETER VALIDATION ============
    type_ = request.args.get("type", "").lower()
    if type_ not in ["real", "ia"]:
        return jsonify({"error": "Invalid type. Must be 'real' or 'ia'"}), 400
    
    variable = request.args.get("variable", "Temperature (°C)")
    
    # Validate coordinates
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing lat/lon parameters"}), 400
    
    is_valid, error_msg = validate_coordinates(lat, lon)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    # ============ REAL-TIME MODE (Sin cambios) ============
    if type_ == "real":
        # Try METAR first for real-time observations
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
        
        # Fallback to forecast APIs
        days = request.args.get("days", default=7, type=int)
        days = max(1, min(days, 16))
        
        df, meta = fetch_hourly_weatherapi(lat, lon, variable, days)
        if df.empty:
            df, meta = fetch_hourly_openmeteo(lat, lon, variable, days)
        if df.empty:
            df, meta = fetch_hourly_nasa(lat, lon, variable, days)
        
        if df.empty:
            return jsonify({"error": "No real-time data available from any source"}), 404
        
        return jsonify({
            "metadata": meta,
            "data": df.to_dict(orient="records"),
            "count": len(df)
        })
    
    # ============ AI SCIENTIFIC MODE ============
    elif type_ == "ia":
        date_str = request.args.get("date")
        if not date_str:
            return jsonify({"error": "Missing 'date' parameter for AI mode"}), 400
        
        try:
            target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        today = datetime.date.today()
        is_future = target_date > today
        is_past = target_date < today
        
        years_back = request.args.get("years", default=10, type=int)
        years_back = max(1, min(years_back, 30))
        
        # Define historical data range
        start_hist = today - datetime.timedelta(days=365 * years_back)
        end_hist = today - datetime.timedelta(days=1)  # Up to yesterday
        
        # ============ FETCH DAILY HISTORICAL DATA ============
        print(f"📊 Fetching daily historical data from {start_hist} to {end_hist}")
        
        hist_df = fetch_meteostat_daily(lat, lon, start_hist, end_hist, variable)
        if hist_df.empty:
            hist_df = fetch_nasa_daily(lat, lon, start_hist, end_hist, variable)
        if hist_df.empty:
            hist_df = fetch_visualcrossing_daily(lat, lon, start_hist, end_hist, variable)
        
        if hist_df.empty:
            return jsonify({"error": "No historical data available for analysis"}), 404
        
        # ============ CASE 1: PAST DATE ============
        if is_past:
            observed_value = get_historical_day_value(hist_df, target_date)
            
            if observed_value is not None:
                # Return actual observed value
                return jsonify({
                    "date": date_str,
                    "type": "historical_observation",
                    "value": round(observed_value, 2),
                    "source": "Historical records",
                    "variable": variable,
                    "metadata": {
                        "data_source": "Meteostat/NASA POWER/Visual Crossing",
                        "observation_type": "daily_average"
                    },
                    "historical_context": {
                        "available_data_points": len(hist_df),
                        "data_range": f"{hist_df['date'].min()} to {hist_df['date'].max()}"
                    }
                })
            else:
                return jsonify({
                    "error": f"No historical data available for {date_str}",
                    "available_range": f"{hist_df['date'].min()} to {hist_df['date'].max()}"
                }), 404
        
        # ============ CASE 2: FUTURE DATE ============
        elif is_future:
            predicted_value, forecast_df = train_predict_model(hist_df, target_date)
            
            if predicted_value is None:
                return jsonify({"error": "Unable to generate prediction"}), 500
            
            # Calculate prediction interval
            prediction_lower = None
            prediction_upper = None
            if not forecast_df.empty and 'yhat_lower' in forecast_df.columns:
                prediction_lower = float(forecast_df['yhat_lower'].iloc[0])
                prediction_upper = float(forecast_df['yhat_upper'].iloc[0])
            
            return jsonify({
                "date": date_str,
                "type": "ai_prediction",
                "predicted_value": round(predicted_value, 2),
                "prediction_interval": {
                    "lower": round(prediction_lower, 2) if prediction_lower else None,
                    "upper": round(prediction_upper, 2) if prediction_upper else None,
                    "confidence": "80%"
                },
                "model": "Prophet (Meta/Facebook)",
                "variable": variable,
                "training_metadata": {
                    "training_samples": len(hist_df),
                    "training_period_years": years_back,
                    "data_range": f"{hist_df['date'].min()} to {hist_df['date'].max()}",
                    "model_components": ["trend", "weekly_seasonality", "yearly_seasonality"]
                },
                "historical_statistics": {
                    "mean": round(hist_df['value'].mean(), 2),
                    "std": round(hist_df['value'].std(), 2),
                    "min": round(hist_df['value'].min(), 2),
                    "max": round(hist_df['value'].max(), 2)
                }
            })
        
        # ============ CASE 3: TODAY ============
        else:  # target_date == today
            return jsonify({
                "error": "For today's data, use type='real' instead",
                "suggestion": f"/api/forecast?type=real&variable={variable}&lat={lat}&lon={lon}"
            }), 400

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "apis_configured": {
            "weatherapi": bool(WEATHERAPI_KEY),
            "visualcrossing": bool(VISUALCROSSING_KEY)
        }
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
    print("🌦️  Weather Forecast API Starting...")
    print(f"📍 NASA POWER: ✓")
    print(f"📍 Open-Meteo: ✓")
    print(f"📍 WeatherAPI: {'✓' if WEATHERAPI_KEY else '✗ (not configured)'}")
    print(f"📍 Visual Crossing: {'✓' if VISUALCROSSING_KEY else '✗ (not configured)'}")
    print("\n📊 Mode Configuration:")
    print("   • REAL mode: Hourly forecasts (next 7-16 days)")
    print("   • IA mode: Daily analysis (past observations + future predictions)")
    app.run(debug=True, port=5000, host="0.0.0.0")
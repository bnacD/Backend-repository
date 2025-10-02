from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from meteostat import Point, Daily
from prophet import Prophet
import datetime
import pandas as pd
import requests
import re
import math

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Keys
WEATHERAPI_KEY = "60a4eb6709374ba2b5e32318250210"
VISUALCROSSING_KEY = "66G6M8FZPJ3F3G9TDGFG2ZHRQ"

# ---------------- METAR SGAS ----------------
def fetch_metar_sgas():
    """METAR de Silvio Pettirossi, solo si es reciente."""
    try:
        url = "https://tgftp.nws.noaa.gov/data/observations/metar/stations/SGAS.TXT"
        r = requests.get(url, timeout=5)
        r.raise_for_status()

        lines = r.text.strip().split("\n")
        fecha_obs_str = lines[0].strip()  # ej: 2025/10/02 18:00
        metar_line = lines[1].strip()

        # Verificar frescura (menos de 90 minutos)
        fecha_obs = datetime.datetime.strptime(fecha_obs_str, "%Y/%m/%d %H:%M")
        if (datetime.datetime.utcnow() - fecha_obs).total_seconds() > 5400:
            return {}

        # Temp/dewpoint
        temp_match = re.search(r"(\d{2})/(\d{2})", metar_line)
        temperatura_C = int(temp_match.group(1)) if temp_match else None
        dewpoint_C = int(temp_match.group(2)) if temp_match else None

        # Humedad relativa
        humedad_pct = None
        if temperatura_C is not None and dewpoint_C is not None:
            es = 6.11 * math.exp(17.62 * temperatura_C / (243.12 + temperatura_C))
            e = 6.11 * math.exp(17.62 * dewpoint_C / (243.12 + dewpoint_C))
            humedad_pct = round((e / es) * 100)

        # Viento
        viento_match = re.search(r"(\d{3})(\d{2})KT", metar_line)
        viento_mps = None
        if viento_match:
            velocidad_nudos = int(viento_match.group(2))
            viento_mps = round(velocidad_nudos * 0.514444)

        # Presión
        presion_match = re.search(r"Q(\d{4})", metar_line)
        presion_hPa = int(presion_match.group(1)) if presion_match else None

        return {
            "source": "DINAC / METAR SGAS (NOAA feed)",
            "fecha_obs": fecha_obs_str,
            "temperatura_C": temperatura_C,
            "humedad_pct": humedad_pct,
            "viento_mps": viento_mps,
            "presion_hPa": presion_hPa
        }
    except Exception as e:
        print("[ERROR METAR SGAS]", e)
        return {}

# ---------------- GENERAL HELPERS ----------------
def clean_invalid_values(df):
    invalid_values = [999, -9999, -999]
    df = df[~df['value'].isin(invalid_values)]
    df = df[df['value'] >= 0]
    return df

def smooth_series(df):
    """Aplica promedio móvil para suavizar datos modelados."""
    if not df.empty and 'value' in df.columns:
        df['value'] = df['value'].rolling(window=3, min_periods=1).mean()
    return df

# ---------------- APIs HORARIAS ----------------
def fetch_hourly_nasa(lat, lon, variable, days_ahead=7):
    var_map = {
        "Temperature (°C)": "T2M",
        "Precipitation (mm)": "PRECTOTCORR",
        "Wind speed (m/s)": "WS10M",
        "Humidity (%)": "RH2M"
    }
    param = var_map.get(variable)
    end_date = datetime.date.today() + datetime.timedelta(days=days_ahead - 1)
    start_date = datetime.date.today()
    try:
        url = (f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
               f"parameters={param}&community=RE&longitude={lon}&latitude={lat}&start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}&format=JSON")
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json().get("properties", {}).get("parameter", {}).get(param, {})
        df = pd.DataFrame([{"date": pd.to_datetime(ts), "value": val} for ts, val in data.items()])
        return smooth_series(df), {"units": "metric", "source": "NASA POWER (modelled data)"}
    except Exception as e:
        print("[ERROR NASA POWER hourly]", e)
        return pd.DataFrame(), {}

def fetch_hourly_opemeteo(lat, lon, variable, days_ahead=7):
    param_map = {
        "Temperature (°C)": "temperature_2m",
        "Precipitation (mm)": "precipitation",
        "Wind speed (m/s)": "windspeed_10m",
        "Humidity (%)": "relativehumidity_2m"
    }
    parameter = param_map.get(variable)
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}&hourly={parameter}&timezone=auto&forecast_days={days_ahead}")
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json().get("hourly", {})
        if not data or parameter not in data:
            return pd.DataFrame(), {}
        df = pd.DataFrame({"date": pd.to_datetime(data["time"]), "value": data[parameter]})
        return smooth_series(df), {"units": "metric", "source": "Open-Meteo (modelled data)"}
    except Exception as e:
        print("[ERROR Open-Meteo]", e)
        return pd.DataFrame(), {}

def fetch_hourly_weatherapi(lat, lon, variable, days_ahead=7):
    var_map_api = {
        "Temperature (°C)": ("temp_c", "°C"),
        "Precipitation (mm)": ("precip_mm", "mm"),
        "Wind speed (m/s)": ("wind_kph", "km/h"),
        "Humidity (%)": ("humidity", "%")
    }
    code, unit = var_map_api.get(variable)
    try:
        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHERAPI_KEY}&q={lat},{lon}&days={days_ahead}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        forecast = r.json().get("forecast", {}).get("forecastday", [])
        records = []
        for day in forecast:
            for hour in day["hour"]:
                records.append({"date": pd.to_datetime(hour["time"]), "value": hour[code]})
        df = pd.DataFrame(records)
        return smooth_series(df), {"units": unit, "source": "WeatherAPI (modelled/obs mix)"}
    except Exception as e:
        print("[ERROR WeatherAPI]", e)
        return pd.DataFrame(), {}

# ---------------- HISTÓRICO ----------------
def fetch_meteostat_data(lat, lon, start, end, variable):
    col_map = {
        "Temperature (°C)": "tavg",
        "Precipitation (mm)": "prcp",
        "Wind speed (m/s)": "wspd",
        "Humidity (%)": "rhum"
    }
    try:
        station_point = Point(lat, lon)
        df = Daily(station_point, start=start, end=end).fetch()
        df['value'] = df[col_map.get(variable)]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df.dropna(subset=['value'], inplace=True)
        return clean_invalid_values(df)
    except Exception as e:
        print("[ERROR Meteostat]", e)
        return pd.DataFrame()

def fetch_visualcrossing(lat, lon, start, end, variable):
    col_map = {
        "Temperature (°C)": "temp",
        "Precipitation (mm)": "precip",
        "Wind speed (m/s)": "windspeed",
        "Humidity (%)": "humidity"
    }
    try:
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start}/{end}"
        params = {"unitGroup": "metric", "key": VISUALCROSSING_KEY, "include": "days"}
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json().get("days", [])
        df = pd.DataFrame([{"date": d["datetime"], "value": d[col_map.get(variable)]} for d in data])
        return df
    except Exception as e:
        print("[ERROR VisualCrossing]", e)
        return pd.DataFrame()

def fetch_nasa_daily(lat, lon, start, end, variable):
    var_map = {
        "Temperature (°C)": "T2M",
        "Precipitation (mm)": "PRECTOTCORR",
        "Wind speed (m/s)": "WS10M",
        "Humidity (%)": "RH2M"
    }
    param = var_map.get(variable)
    try:
        url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
               f"parameters={param}&community=RE&longitude={lon}&latitude={lat}&start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}&format=JSON")
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json().get("properties", {}).get("parameter", {}).get(param, {})
        df = pd.DataFrame([{"date": k, "value": v} for k, v in data.items()])
        return df
    except Exception as e:
        print("[ERROR NASA POWER daily]", e)
        return pd.DataFrame()

# ---------------- PREDICCIÓN ----------------
def train_predict_model(hist_df, predict_dates):
    try:
        hist_df = hist_df.rename(columns={'date': 'ds', 'value': 'y'})
        hist_df['ds'] = pd.to_datetime(hist_df['ds'])
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(hist_df)
        future = pd.DataFrame({'ds': predict_dates})
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'value'})
    except Exception as e:
        print("[ERROR Prophet]", e)
        return pd.DataFrame()

# ---------------- API ----------------
@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    type_ = request.args.get("type", "").lower()
    variable = request.args.get("variable", "Temperature (°C)")
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    fmt = request.args.get("format", "json").lower()

    if type_ == "real":
        if any(k in variable for k in ["Temperature", "Wind", "Humidity", "Pressure"]):
            metar = fetch_metar_sgas()
            if metar:
                valor = None
                unidad = None
                if "Temperature" in variable:
                    valor = metar["temperatura_C"]; unidad = "°C"
                elif "Wind" in variable:
                    valor = metar["viento_mps"]; unidad = "m/s"
                elif "Humidity" in variable:
                    valor = metar["humedad_pct"]; unidad = "%"
                elif "Pressure" in variable:
                    valor = metar["presion_hPa"]; unidad = "hPa"
                if valor is not None:
                    return jsonify({"fecha": metar["fecha_obs"], "valor": valor, "unidad": unidad, "source": metar["source"]})

        # Meteostat primero para series
        start = datetime.datetime.today()
        end = start + datetime.timedelta(days=request.args.get("days", default=7, type=int))
        df = fetch_meteostat_data(lat, lon, start, end, variable)
        meta = {"units": "metric", "source": "Meteostat (observed data)"}
        if df.empty:
            df, meta = fetch_hourly_nasa(lat, lon, variable)
        if df.empty:
            df, meta = fetch_hourly_opemeteo(lat, lon, variable)
        if df.empty:
            df, meta = fetch_hourly_weatherapi(lat, lon, variable)
        if df.empty:
            return jsonify({"error": "No real data available"})
        return jsonify({"metadata": meta, "data": df.to_dict(orient="records")})

    elif type_ == "ia":
        years_back = request.args.get("years", default=10, type=int)
        date_str = request.args.get("date")
        if not date_str:
            return jsonify({"error": "Missing date parameter for IA mode"})
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        start_hist = datetime.datetime.combine(datetime.date.today() - datetime.timedelta(days=365 * years_back), datetime.time.min)
        end_hist = datetime.datetime.combine(datetime.date.today(), datetime.time.min)

        hist_df = fetch_meteostat_data(lat, lon, start_hist, end_hist, variable)
        if hist_df.empty:
            hist_df = fetch_nasa_daily(lat, lon, start_hist.date(), end_hist.date(), variable)
        if hist_df.empty:
            hist_df = fetch_visualcrossing(lat, lon, start_hist.date(), end_hist.date(), variable)
        if hist_df.empty:
            return jsonify({"error": "No historical data available"})

        pred_df = train_predict_model(hist_df, [pd.to_datetime(date_obj)])
        if pred_df.empty:
            return jsonify({"error": "Unable to generate prediction"})
        val = pred_df['value'].iloc[0]
        return jsonify({
            "date": date_str,
            "predicted_value": round(float(val), 2),
            "source": "Prophet + Historical",
            "historical_data": hist_df.to_dict(orient="records")
        })

    return jsonify({"error": "Invalid type parameter"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
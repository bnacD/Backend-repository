from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from meteostat import Point, Daily
from prophet import Prophet
import datetime
import pandas as pd
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# API Keys
VISUALCROSSING_KEY = "66G6M8FZPJ3F3G9TDGFG2ZHRQ"

# ---------------- HELPERS ----------------
def clean_invalid_values(df):
    invalid_values = [999, -9999, -999]
    df = df[~df['value'].isin(invalid_values)]
    df = df[df['value'] >= 0]
    return df

# ---------------- OPEN-METEO (REAL) ----------------
def fetch_hourly_opemeteo(lat, lon, variable, days_ahead=7):
    param_map = {
        "Temperature (°C)": "temperature_2m",
        "Precipitation (mm)": "precipitation",
        "Wind speed (m/s)": "windspeed_10m",
        "Humidity (%)": "relativehumidity_2m"
    }
    parameter = param_map.get(variable, "temperature_2m")
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}&hourly={parameter}&timezone=auto&forecast_days={days_ahead}")
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json().get("hourly", {})
        if not data or parameter not in data:
            return pd.DataFrame(), {}
        df = pd.DataFrame({"date": pd.to_datetime(data["time"]),
                           "value": data[parameter]})
        return df, {"units": "metric", "source": "Open-Meteo forecast"}
    except Exception as e:
        print("[ERROR Open-Meteo]", e)
        return pd.DataFrame(), {}

# ---------------- HISTORICAL SOURCES ----------------
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
        df['value'] = df[col_map.get(variable, "tavg")]
        df = df.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        df.dropna(subset=['value'], inplace=True)
        return clean_invalid_values(df)
    except Exception as e:
        print("[ERROR Meteostat]", e)
        return pd.DataFrame()

def fetch_nasa_daily(lat, lon, start, end, variable):
    var_map = {
        "Temperature (°C)": "T2M",
        "Precipitation (mm)": "PRECTOTCORR",
        "Wind speed (m/s)": "WS10M",
        "Humidity (%)": "RH2M"
    }
    param = var_map.get(variable, "T2M")
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

# ---------------- PREDICTION ----------------
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

# ---------------- API ENDPOINT ----------------
@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    type_ = request.args.get("type", "").lower()
    variable = request.args.get("variable", "Temperature (°C)")
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)
    fmt = request.args.get("format", "json").lower()

    if type_ == "real":
        days_ahead = request.args.get("days", default=7, type=int)
        df, meta = fetch_hourly_opemeteo(lat, lon, variable, days_ahead)
        if df.empty:
            return jsonify({"error": "No forecast data available from Open-Meteo"})
        if fmt == "csv":
            return Response(df.to_csv(index=False), mimetype="text/csv")
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

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from meteostat import Point, Daily
from prophet import Prophet
import datetime
import pandas as pd
import numpy as np
import requests

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# ---------------- HELPER FUNCTIONS ----------------
def clean_invalid_values(df, variable):
    invalid_values = [999, -9999, -999]
    df = df[~df['value'].isin(invalid_values)]
    if variable in ["Precipitation (mm)", "Wind speed (m/s)"]:
        df = df[df['value'] >= 0]
    return df

def fetch_meteostat_data(lat, lon, start_date, end_date, variable):
    """Fetch historical data from Meteostat."""
    try:
        station_point = Point(lat, lon)
        data = Daily(station_point, start=start_date, end=end_date).fetch()
        col_map = {
            "Temperature (°C)": "tavg",
            "Precipitation (mm)": "prcp",
            "Wind speed (m/s)": "wspd"
        }
        data['value'] = data[col_map.get(variable, "tavg")]
        data = data.reset_index()[['time', 'value']].rename(columns={'time': 'date'})
        data.dropna(subset=['value'], inplace=True)
        return clean_invalid_values(data, variable)
    except Exception as e:
        print(f"[ERROR Meteostat] {e}")
        return pd.DataFrame(columns=['date', 'value'])

def fetch_hourly_forecast(lat, lon, variable):
    """Fetch hourly forecast data from Open-Meteo."""
    try:
        param_map = {
            "Temperature (°C)": "temperature_2m",
            "Precipitation (mm)": "precipitation",
            "Wind speed (m/s)": "windspeed_10m"
        }
        parameter = param_map.get(variable, "temperature_2m")
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly={parameter}&timezone=auto&forecast_days=14"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("hourly", {})
        if not data or parameter not in data:
            return pd.DataFrame(columns=["date", "value"])
        return pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "value": data[parameter]
        })
    except Exception as e:
        print(f"[ERROR Open-Meteo] {e}")
        return pd.DataFrame(columns=["date", "value"])

def train_predict_model(hist_df, predict_dates):
    """Train Prophet model and predict future values."""
    try:
        if hist_df.empty:
            return pd.DataFrame(columns=['date','value'])
        hist_df = hist_df.rename(columns={'date':'ds','value':'y'})
        hist_df['ds'] = pd.to_datetime(hist_df['ds'])
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(hist_df)
        future = pd.DataFrame({'ds': predict_dates})
        forecast = model.predict(future)
        return forecast[['ds','yhat']].rename(columns={'ds':'date','yhat':'value'})
    except Exception as e:
        print(f"[ERROR Prophet] {e}")
        return pd.DataFrame(columns=['date', 'value'])

def climate_sensation(variable, val):
    """Return qualitative climate sensation."""
    if variable == "Temperature (°C)":
        return "Cold" if val < 18 else "Hot" if val > 30 else "Pleasant"
    elif variable == "Precipitation (mm)":
        return "Rainy" if val > 2 else "Dry"
    elif variable == "Wind speed (m/s)":
        return "Windy" if val > 5 else "Calm"
    return ""

# ---------------- API ROUTE ----------------
@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    """Main API endpoint for weather forecast."""
    try:
        type_ = request.args.get("type")  # 'real' or 'ia'
        variable = request.args.get("variable", "Temperature (°C)")
        lat = float(request.args.get("lat", -25.2637))
        lon = float(request.args.get("lon", -57.5759))
        date_str = request.args.get("date", datetime.date.today().strftime("%Y-%m-%d"))
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

        if type_ == "real":
            df_hourly = fetch_hourly_forecast(lat, lon, variable)
            df_day = df_hourly[df_hourly['date'].dt.date == date_obj]
            if df_day.empty:
                return jsonify({"error": "No hourly data available for the selected date"})
            val = df_day['value'].max() if variable != "Precipitation (mm)" else df_day['value'].sum()
            return jsonify({
                "date": date_str,
                "value": round(val, 2),
                "feeling": climate_sensation(variable, val)
            })

        elif type_ == "ia":
            years_back = int(request.args.get("years", 10))
            start_hist = datetime.datetime.combine(datetime.date.today() - datetime.timedelta(days=365 * years_back), datetime.time.min)
            end_hist = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
            hist_df = fetch_meteostat_data(lat, lon, start_hist, end_hist, variable)
            if hist_df.empty:
                return jsonify({"error": "No historical data found for the selected location"})
            pred_df = train_predict_model(hist_df, [pd.to_datetime(date_obj)])
            val = pred_df['value'].iloc[0]
            return jsonify({
                "date": date_str,
                "value": round(float(val), 2),
                "feeling": climate_sensation(variable, val)
            })

        return jsonify({"error": "Invalid type parameter"})
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from meteostat import Point, Daily
from prophet import Prophet
import datetime
import pandas as pd
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------- Helpers --------------------
def clean_invalid_values(df, variable):
    invalid_values = [999, -9999, -999]
    df = df[~df['value'].isin(invalid_values)]
    if variable in ["Precipitation (mm)", "Wind speed (m/s)", "Humidity (%)"]:
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
            "Wind speed (m/s)": "wspd",
            "Humidity (%)": "rhum"
        }
        data["value"] = data[col_map.get(variable, "tavg")]
        data = data.reset_index()[["time", "value"]].rename(columns={"time": "date"})
        data.dropna(subset=["value"], inplace=True)
        return clean_invalid_values(data, variable)
    except Exception as e:
        print(f"[ERROR Meteostat] {e}")
        return pd.DataFrame(columns=["date", "value"])


def fetch_hourly_forecast(lat, lon, variable, days_ahead=7):
    """Fetch hourly forecast from Open-Meteo."""
    param_map = {
        "Temperature (°C)": "temperature_2m",
        "Precipitation (mm)": "precipitation",
        "Wind speed (m/s)": "windspeed_10m",
        "Humidity (%)": "relativehumidity_2m"
    }
    parameter = param_map.get(variable, "temperature_2m")

    if days_ahead < 1 or days_ahead > 16:
        days_ahead = 7  # default to 7 days if invalid

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly={parameter}&timezone=auto&forecast_days={days_ahead}"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("hourly", {})
        if not data or parameter not in data:
            return pd.DataFrame(columns=["date", "value"]), {}
        df = pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "value": data[parameter]
        })
        meta = {"units": "units unknown", "source": "https://api.open-meteo.com"}
        return df, meta
    except Exception as e:
        print(f"[ERROR Open-Meteo] {e}")
        return pd.DataFrame(columns=["date", "value"]), {}


def train_predict_model(hist_df, predict_dates):
    """Train Prophet AI model."""
    try:
        if hist_df.empty:
            return pd.DataFrame(columns=["date", "value"])
        hist_df = hist_df.rename(columns={"date": "ds", "value": "y"})
        hist_df["ds"] = pd.to_datetime(hist_df["ds"])
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(hist_df)
        future = pd.DataFrame({"ds": predict_dates})
        forecast = model.predict(future)
        return forecast[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "value"})
    except Exception as e:
        print(f"[ERROR Prophet] {e}")
        return pd.DataFrame(columns=["date", "value"])


def climate_sensation(variable, val):
    """Return qualitative feeling."""
    if variable == "Temperature (°C)":
        return "Cold" if val < 18 else "Hot" if val > 30 else "Pleasant"
    elif variable == "Precipitation (mm)":
        return "Rainy" if val > 2 else "Dry"
    elif variable == "Wind speed (m/s)":
        return "Windy" if val > 5 else "Calm"
    elif variable == "Humidity (%)":
        return "Humid" if val > 70 else "Comfortable"
    return ""


# -------------------- Endpoint --------------------
@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    try:
        type_ = request.args.get("type", "").lower()
        variable = request.args.get("variable", "Temperature (°C)")
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)
        fmt = request.args.get("format", "json").lower()

        # Mode: REAL (multiple days ahead)
        if type_ == "real":
            days_ahead = request.args.get("days", default=7, type=int)
            df_hourly, meta = fetch_hourly_forecast(lat, lon, variable, days_ahead)
            if df_hourly.empty:
                return jsonify({"error": "No hourly data available"})
            output = {
                "metadata": meta,
                "data": df_hourly.to_dict(orient="records")
            }
            if fmt == "csv":
                return Response(df_hourly.to_csv(index=False), mimetype="text/csv")
            return jsonify(output)

        # Mode: AI Historical
        elif type_ == "ia":
            date_str = request.args.get("date")
            if not date_str:
                return jsonify({"error": "Missing date parameter for AI mode"})
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            years_back = request.args.get("years", default=10, type=int)
            start_hist = datetime.datetime.combine(datetime.date.today() - datetime.timedelta(days=365 * years_back), datetime.time.min)
            end_hist = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
            hist_df = fetch_meteostat_data(lat, lon, start_hist, end_hist, variable)
            if hist_df.empty:
                return jsonify({"error": "No historical data found"})
            pred_df = train_predict_model(hist_df, [pd.to_datetime(date_obj)])
            val = pred_df['value'].iloc[0]
            output = {
                "date": date_str,
                "predicted_value": round(float(val), 2),
                "feeling": climate_sensation(variable, val),
                "historical_data": hist_df.to_dict(orient="records")
            }
            if fmt == "csv":
                return Response(pred_df.to_csv(index=False), mimetype="text/csv")
            return jsonify(output)

        # Invalid type
        return jsonify({"error": "Invalid type parameter"})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
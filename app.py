import os
import gdown
import joblib
import dagshub
import mlflow
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from time import sleep

# Streamlit page config
st.set_page_config(
    page_title="Uber Demand Prediction",
    page_icon="🚕🌆",
    layout="wide"
)

# Styling with HTML/CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #ece9e6, #ffffff);
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #333333;
        text-align: center;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5em 1em;
        border-radius: 10px;
        border: none;
    }
    .stProgress > div > div {
        background-color: #4CAF50 !important;
    }
    .legend-box {
        background-color: #f9f9f9;
        padding: 1em;
        border-radius: 10px;
        margin-top: 1em;
        box-shadow: 0px 0px 10px #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize DAGsHub + MLflow
dagshub.init(repo_owner='MohammoD2', repo_name='Uber_Demand_Pridiction-', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/MohammoD2/Uber_Demand_Pridiction-.mlflow")

# Load model from registry
registered_model_name = 'uber_demand_prediction_model'
stage = "Production"
model_registry_path = f"models:/{registered_model_name}/{stage}"
reg_model = mlflow.sklearn.load_model(model_registry_path)

# Google Drive secured joblib file loading
GDRIVE_FILES = {
    "scaler": os.environ["SCALER_KEY"],
    "encoder": os.environ["ENCODER_KEY"],
    "kmeans": os.environ["KMEANS_KEY"],
    "model": os.environ["MODEL_KEY"]
}
LOCAL_FILES = {
    "scaler": "models/scaler.joblib",
    "encoder": "models/encoder.joblib",
    "kmeans": "models/mb_kmeans.joblib",
    "model": "models/model.joblib"
}

# Download from Google Drive
def download_file_from_drive(file_id, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {destination}..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", destination, quiet=False)

for key, file_id in GDRIVE_FILES.items():
    download_file_from_drive(file_id, LOCAL_FILES[key])

# Load joblib models
scaler = joblib.load(LOCAL_FILES["scaler"])
encoder = joblib.load(LOCAL_FILES["encoder"])
model = joblib.load(LOCAL_FILES["model"])
kmeans = joblib.load(LOCAL_FILES["kmeans"])

# Load data
root_path = Path(__file__).parent
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# UI Title
st.title("Uber Demand in New York City 🚕🌆")

# Sidebar map type
st.sidebar.title("Options")
map_type = st.sidebar.radio("Select Map Type", ["Complete NYC Map", "Only for Neighborhood Regions"], index=1)

# Select date and time
st.subheader("Date")
date = st.date_input("Select the date", value=dt.date(2016, 3, 1), min_value=dt.date(2016, 3, 1), max_value=dt.date(2016, 3, 31))
st.subheader("Time")
time = st.time_input("Select the time")

if date and time:
    delta = dt.timedelta(minutes=15)
    next_interval = dt.datetime.combine(date, time) + delta
    index = pd.Timestamp(next_interval)

    st.markdown(f"**Prediction Time:** {index}")

    sample_loc = df_plot.sample(1).reset_index(drop=True)
    lat, long, region = sample_loc["pickup_latitude"].item(), sample_loc["pickup_longitude"].item(), sample_loc["region"].item()
    st.markdown(f"**Location**: Lat {lat}, Long {long}, Region ID: {region}")

    scaled_cord = scaler.transform(sample_loc.iloc[:, :2])

    st.subheader("Map")
    colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", "#32CD32", "#008000", "#006400",
              "#00FF00", "#7CFC00", "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", "#0000FF",
              "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", "#FF00FF", "#FF1493", "#C71585", "#FF6347",
              "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA", "#DAA520"]

    region_colors = {r: colors[i % len(colors)] for i, r in enumerate(df_plot["region"].unique())}
    df_plot["color"] = df_plot["region"].map(region_colors)

    pipe = Pipeline([
        ("encoder", encoder),
        ("reg", model)
    ])

    if map_type == "Complete NYC Map":
        progress_bar = st.progress(0, text="Loading Complete NYC Map")
        for pct in range(100):
            sleep(0.01)
            progress_bar.progress(pct + 1, text="Rendering Map...")
        st.map(data=df_plot, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
        progress_bar.empty()

        input_data = df.loc[index, :].sort_values("region")
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("<div class='legend-box'><h3>Map Legend</h3>", unsafe_allow_html=True)
        for i in range(len(predictions)):
            rid = input_data.iloc[i]["region"]
            col = region_colors[rid]
            st.markdown(f"<div style='display:flex;align-items:center;'><div style='background:{col};width:20px;height:10px;margin-right:10px;'></div>Region {rid} - Demand: {int(predictions[i])}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif map_type == "Only for Neighborhood Regions":
        distances = kmeans.transform(scaled_cord).ravel()
        close_indexes = sorted(range(len(distances)), key=lambda i: distances[i])[:9]
        df_plot_filtered = df_plot[df_plot["region"].isin(close_indexes)]

        progress_bar = st.progress(0, text="Loading Neighborhood Map")
        for pct in range(100):
            sleep(0.01)
            progress_bar.progress(pct + 1, text="Rendering Map...")
        st.map(data=df_plot_filtered, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
        progress_bar.empty()

        input_data = df.loc[index, :]
        input_data = input_data[input_data["region"].isin(close_indexes)].sort_values("region")
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("<div class='legend-box'><h3>Map Legend</h3>", unsafe_allow_html=True)
        for i in range(len(predictions)):
            rid = input_data.iloc[i]["region"]
            col = region_colors[rid]
            st.markdown(f"<div style='display:flex;align-items:center;'><div style='background:{col};width:20px;height:10px;margin-right:10px;'></div>Region {rid} - Demand: {int(predictions[i])}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<hr><center>Built with ❤️ by <a href='https://github.com/MohammoD2' target='_blank'>MohammoD2</a></center>", unsafe_allow_html=True)

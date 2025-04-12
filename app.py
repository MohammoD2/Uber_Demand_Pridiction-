import streamlit as st
import pandas as pd
import datetime as dt
import joblib
import mlflow
import dagshub
from pathlib import Path
from sklearn.pipeline import Pipeline
from time import sleep

# Page config
st.set_page_config(page_title="Uber Demand Prediction", page_icon="🌆")

# DagsHub MLflow authorization
dagshub.init(repo_owner=st.secrets["DAGSHUB"]["USERNAME"],
             repo_name='Uber_Demand_Pridiction-',
             mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/MohammoD2/Uber_Demand_Pridiction-.mlflow")

# Model name and version
registered_model_name = 'uber_demand_prediction_model'
stage = "Production"
model_path_registry = f"models:/{registered_model_name}/{stage}"

# Load model from MLflow
mlflow_model = mlflow.sklearn.load_model(model_path_registry)

# Google Drive file keys (from secrets)
SCALER_KEY = st.secrets["GDRIVE_KEYS"]["SCALER_KEY"]
ENCODER_KEY = st.secrets["GDRIVE_KEYS"]["ENCODER_KEY"]
KMEANS_KEY = st.secrets["GDRIVE_KEYS"]["KMEANS_KEY"]
MODEL_KEY = st.secrets["GDRIVE_KEYS"]["MODEL_KEY"]
TEST_CSV = st.secrets["GDRIVE_KEYS"]["TEST_CSV"]
PLOT_DATA = st.secrets["GDRIVE_KEYS"]["PLOT_DATA"]

# Helper to download from GDrive
@st.cache_data
def download_from_drive(file_id, filename):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, filename, quiet=False)
    return filename

# Download models/data from Google Drive
scaler_path = download_from_drive(SCALER_KEY, "scaler.joblib")
encoder_path = download_from_drive(ENCODER_KEY, "encoder.joblib")
model_path = download_from_drive(MODEL_KEY, "model.joblib")
plot_data_path = download_from_drive(PLOT_DATA, "plot_data.csv")
test_data_path = download_from_drive(TEST_CSV, "test.csv")

# Load assets
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(test_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# UI
st.title("Uber Demand in New York City 🚕🌆")

st.sidebar.title("Options")
map_type = st.sidebar.radio(label="Select the type of Map",
                            options=["Complete NYC Map"],
                            index=0)

# Date selection
st.subheader("Date")
date = st.date_input("Select the date", value=None,
                     min_value=dt.date(2016, 3, 1),
                     max_value=dt.date(2016, 3, 31)) 
st.write("**Date:**", date)

# Time selection
st.subheader("Time")
time = st.time_input("Select the time", value=None)
st.write("**Current Time:**", time)

if date and time:
    # Calculate next 15-minute interval
    delta = dt.timedelta(minutes=15)
    next_interval = dt.datetime.combine(date, time) + delta
    st.write("Demand for Time: ", next_interval.time())

    index = pd.Timestamp(f"{date} {next_interval.time()}")
    st.write("**Date & Time:**", index)

    # Sample random location
    st.subheader("Location")
    sample_loc = df_plot.sample(1).reset_index(drop=True)
    lat = sample_loc["pickup_latitude"].item()
    long = sample_loc["pickup_longitude"].item()
    region = sample_loc["region"].item()

    st.write("**Your Current Location**")
    st.write(f"Lat: {lat}")
    st.write(f"Long: {long}")

    with st.spinner("Fetching your Current Region"):
        sleep(3)
    st.write("Region ID: ", region)

    # Apply scaler
    scaled_cord = scaler.transform(sample_loc.iloc[:, 0:2])

    # Create color mapping
    colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", 
              "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", 
              "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", 
              "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", 
              "#FF00FF", "#FF1493", "#C71585", "#FF6347", "#FFA07A", 
              "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA", "#FFB6C1"]

    region_colors = {region: colors[i] for i, region in enumerate(df_plot["region"].unique())}
    df_plot["color"] = df_plot["region"].map(region_colors)

    # Prediction pipeline
    pipe = Pipeline([
        ('encoder', encoder),
        ('reg', model)
    ])

    # Show complete NYC map
    if map_type == "Complete NYC Map":
        progress_bar = st.progress(value=0, text="Loading map...")
        for i in range(100):
            sleep(0.01)
            progress_bar.progress(i + 1, text="Loading map...")

        st.map(df_plot, latitude="pickup_latitude", 
               longitude="pickup_longitude", size=0.01,
               color="color")
        progress_bar.empty()

        # Predict demand
        if index in df.index:
            input_data = df.loc[index, :].sort_values("region")
            target = input_data["total_pickups"]
            predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

            # Map Legend
            st.markdown("### Map Legend")
            for i in range(len(predictions)):
                region_id = input_data.iloc[i]["region"]
                demand = int(predictions[i])
                color = region_colors.get(region_id, "#000000")
                label = f"{region_id} (Current region)" if region == region_id else f"{region_id}"
                st.markdown(
                    f'<div style="display: flex; align-items: center;">'
                    f'<div style="background-color:{color}; width: 20px; height: 10px; margin-right: 10px;"></div>'
                    f'Region ID: {label} <br> Demand: {demand} <br><br>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No data for the selected date & time.")

import streamlit as st
import pandas as pd
import datetime as dt
import joblib
import mlflow
import dagshub
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from time import sleep
import folium
from streamlit_folium import folium_static

# Page config
st.set_page_config(page_title="Uber Demand Prediction", page_icon="🌆")

# # Inject DAGSHUB_TOKEN for MLflow authorization
# os.environ["DAGSHUB_TOKEN"] = st.secrets["DAGSHUB"]["TOKEN"]

# # DagsHub MLflow authorization
# dagshub.init(
#     repo_owner=st.secrets["DAGSHUB"]["USERNAME"],
#     repo_name='Uber_Demand_Pridiction-',
#     mlflow=True
# )
# mlflow.set_tracking_uri(f"https://dagshub.com/{st.secrets['DAGSHUB']['USERNAME']}/Uber_Demand_Pridiction-.mlflow")

# # Load model from MLflow
# registered_model_name = 'uber_demand_prediction_model'
# stage = "Production"
# model_path_registry = f"models:/{registered_model_name}/{stage}"

# try:
#     mlflow_model = mlflow.sklearn.load_model(model_path_registry)
#     st.success("✅ Successfully authorized and loaded the ML model from DagsHub.")
# except Exception as e:
#     st.error(f"❌ Failed to load model: {e}")

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

    # Add region names mapping (based on NYC neighborhoods)
    REGION_NAMES = {
        0: "Upper West Side",
        1: "Upper East Side",
        2: "Midtown West",
        3: "Midtown East",
        4: "Chelsea",
        5: "Gramercy",
        6: "Greenwich Village",
        7: "SoHo",
        8: "Tribeca",
        9: "Financial District",
        10: "East Village",
        11: "Lower East Side",
        12: "East Harlem",
        13: "Central Harlem",
        14: "Morningside Heights",
        15: "Hamilton Heights",
        16: "Washington Heights",
        17: "Inwood",
        18: "Roosevelt Island",
        19: "Battery Park",
        20: "Chinatown",
        21: "NoHo",
        22: "Civic Center",
        23: "Little Italy",
        24: "Nolita",
        25: "Two Bridges",
        26: "Stuyvesant Town",
        27: "Kips Bay",
        28: "Murray Hill",
        29: "Tudor City"
    }

    # Show complete NYC map
    if map_type == "Complete NYC Map":
        progress_bar = st.progress(value=0, text="Loading map...")
        for i in range(100):
            sleep(0.01)
            progress_bar.progress(i + 1, text="Loading map...")

        # Create a Folium map centered on NYC
        m = folium.Map(location=[40.7831, -73.9712], zoom_start=12)

        # Add all regions as circles
        for _, row in df_plot.iterrows():
            color = region_colors.get(row["region"], "#000000")
            folium.CircleMarker(
                location=[row["pickup_latitude"], row["pickup_longitude"]],
                radius=3,
                color=color,
                fill=True,
                popup=f"Region: {REGION_NAMES.get(row['region'], f'Region {row['region']}')}",
            ).add_to(m)

        # Add current location with a special marker
        if 'lat' in locals() and 'long' in locals():
            folium.Marker(
                location=[lat, long],
                popup=f"Your Location<br>Region: {REGION_NAMES.get(region, f'Region {region}')}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)

        # Display the map
        folium_static(m)
        progress_bar.empty()

        # Predict demand
        if index in df.index:
            input_data = df.loc[index, :].sort_values("region")
            target = input_data["total_pickups"]
            predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

            # Enhanced Map Legend with Region Names
            st.markdown("### Region Information")
            
            # Current Region Highlight
            if 'region' in locals():
                st.markdown(f"""
                #### 📍 Your Current Location
                - **Region Name:** {REGION_NAMES.get(region, f'Region {region}')}
                - **Region ID:** {region}
                - **Coordinates:** ({lat:.4f}, {long:.4f})
                """)
                st.markdown("---")

            st.markdown("### All Regions")
            for i in range(len(predictions)):
                region_id = input_data.iloc[i]["region"]
                demand = int(predictions[i])
                color = region_colors.get(region_id, "#000000")
                region_name = REGION_NAMES.get(region_id, f"Region {region_id}")
                is_current = region == region_id
                
                st.markdown(
                    f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                    f'<div style="background-color:{color}; width: 20px; height: 20px; margin-right: 10px; border-radius: 50%;"></div>'
                    f'<div><strong>{region_name}</strong> {" (Current Location)" if is_current else ""}<br>'
                    f'Region ID: {region_id}<br>'
                    f'Predicted Demand: {demand}</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No data for the selected date & time.")
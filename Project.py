import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import folium
from folium.plugins import TimestampedGeoJson
import random
import datetime
from shapely.ops import cascaded_union
import streamlit as st
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


st.markdown("<h1 style='text-align: center;'>Internship Project</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>Anomaly Detection in Flight Paths</h1>", unsafe_allow_html=True)


df = pd.read_csv('combined_data.csv')
df['new_time'] = pd.to_datetime(df['time'], unit='s')
df['date1'] = df['new_time'].dt.date
df['time1'] = df['new_time'].dt.time
st.markdown("<h3 style='text-align: center;'>Top 5 Rows of Data</h3>", unsafe_allow_html=True)
st.write(df.head())


n = df['callsign'].nunique()
m = df['icao24'].nunique()
st.markdown(f"<h3 style='text-align: center;'>There are {n} Paths of {m} Planes</h3>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>Data Visualization</h3>", unsafe_allow_html=True)
grouped_flights = df.groupby('callsign')
map_center = [df['lat'].mean(), df['lon'].mean()]
my_map = folium.Map(location=map_center, zoom_start=8)
for callsign, group in grouped_flights:
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    folium.PolyLine(
        locations=group[['lat', 'lon']].values.tolist(),
        color=color,
        weight=2,
        opacity=1,
        popup=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}",
        tooltip=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}"
    ).add_to(my_map)
folium_static(my_map)


l = ["AIC811",'AAR7683','AIC406','AIC401']  #'AIC409'  anomaly deviation  'AIC306''AIC762',
df1 = df[df['callsign'].isin(l)]
grouped_flights = df1.groupby('callsign')


st.markdown("<h3 style='text-align: center;'>Let say this is a common path for multiple flights having same Departure and Arrival Points</h3>", unsafe_allow_html=True)
map_center = [df1['lat'].mean(), df1['lon'].mean()]
my_map = folium.Map(location=map_center, zoom_start=8)

for callsign, group in grouped_flights:
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    folium.PolyLine(
        locations=group[['lat', 'lon']].values.tolist(),
        color=color,
        weight=2,
        opacity=1,
        popup=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}",
        tooltip=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}"
    ).add_to(my_map)
folium_static(my_map)


st.markdown("<h3 style='text-align: center;'>Now we create a Boundary over these Paths Shows as Normal Flight Area for this Route</h3>", unsafe_allow_html=True)
all_buffers = []
for callsign, group in grouped_flights:
    points = [Point(lon, lat) for lon, lat in zip(group['lon'], group['lat'])]
    line = LineString(points)
    buffer_distance = 0.045
    buffer = line.buffer(buffer_distance)    
    all_buffers.append(buffer)
overall_boundary = cascaded_union(all_buffers)
overall_centroid = overall_boundary.centroid

map_center = [overall_centroid.y, overall_centroid.x]
my_map = folium.Map(location=map_center, zoom_start=8)

for callsign, group in grouped_flights:
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    folium.PolyLine(
        locations=group[['lat', 'lon']].values.tolist(),
        color=color,
        popup=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}",
        tooltip=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}"
    ).add_to(my_map)

folium.GeoJson(
    overall_boundary.__geo_interface__,
    style_function=lambda x, color="#0000FF": {'fillColor': color, 'color': 'black', 'fillOpacity': 0.3},
    popup="Overall Boundary",
    tooltip="Overall Boundary"
).add_to(my_map)
folium_static(my_map)


st.markdown("<h3 style='text-align: center;'>Now we check that any new flight on the same path is in the noraml region or not if not then print alert</h3>", unsafe_allow_html=True)
all_buffers = []
for callsign, group in grouped_flights:
    points = [Point(lon, lat) for lon, lat in zip(group['lon'], group['lat'])]    
    line = LineString(points)    
    buffer_distance = 0.045
    buffer = line.buffer(buffer_distance)    
    all_buffers.append(buffer)
overall_boundary = cascaded_union(all_buffers)
overall_centroid = overall_boundary.centroid

map_center = [overall_centroid.y, overall_centroid.x]
my_map = folium.Map(location=map_center, zoom_start=8)

for callsign, group in grouped_flights:
    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    folium.PolyLine(
        locations=group[['lat', 'lon']].values.tolist(),
        color=color,
        popup=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}",
        tooltip=f"Flight: {group['icao24'].iloc[0]}<br>Callsign: {callsign}"
    ).add_to(my_map)

folium.GeoJson(
    overall_boundary.__geo_interface__,
    style_function=lambda x, color="#0000FF": {'fillColor': color, 'color': 'black', 'fillOpacity': 0.3},
    popup="Overall Boundary",
    tooltip="Overall Boundary"
).add_to(my_map)

features = []
z = df[df['callsign'] == 'AIC409']
z['time'] = pd.to_datetime(z['time'], unit='s')

# Check if any point goes outside the boundary during the animation
outside_boundary_alert_printed = False
for _, row in z.iterrows():
    lat, lon = row['lat'], row['lon']
    point = Point(lon, lat)
    
    # Check if the point is outside the overall boundary
    if not overall_boundary.contains(point):
        outside_boundary_alert_printed = True
        break

# Print the alert if any point is outside the boundary
if outside_boundary_alert_printed:
    st.write(f"Alert: Path for callsign AIC409 is outside the overall boundary.")

# Construct the feature for animation
line_coordinates = [[row['lon'], row['lat']] for _, row in z.iterrows()]
feature = {
    'type': 'Feature',
    'geometry': {
        'type': 'LineString',
        'coordinates': line_coordinates
    },
    'properties': {
        'times': [time.strftime('%Y-%m-%dT%H:%M:%S') for time in z['time']]
    },
    'style': {
        'color': "#{:06x}".format(random.randint(0, 0xFFFFFF))  
    }
}
features.append(feature)

# Add TimestampedGeoJson to map with multiple LineStrings
TimestampedGeoJson(
    {'type': 'FeatureCollection',
     'features': features},
    period='PT1S',       
    add_last_point=True
).add_to(my_map)
folium_static(my_map)


st.markdown("<h1 style='text-align: center;'>Flight Trajectory Forecasting</h1>", unsafe_allow_html=True)


l = ["AIC811",'AAR7683','AIC406','AIC401']
path = st.selectbox("Choose a flight you want to Forecast for next 2 Minutes", l)

a = df[df['callsign'] == path]
a = a[['time','lat','lon']]

# Prepare data
X = a[['time']].values  # Predictor variable: 'new_time'
y = a[['lat', 'lon']].values  # Target variables

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Reshape input data for LSTM
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=2))  # Output layer predicts 'lat', 'lon', 'heading', 'velocity', 'baroaltitude'
model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
model.fit(X_train_reshaped, y_train_scaled, epochs=100, batch_size=32, verbose=1)

last_timestamp = X[-1][0] 
timestamps_out_of_sample = [pd.Timestamp.utcfromtimestamp(last_timestamp) + pd.Timedelta(seconds=i) for i in range(120)]

# Convert timestamps to numeric representation
X_out_of_sample = np.array([timestamp.timestamp() for timestamp in timestamps_out_of_sample]).reshape(-1, 1)

# Scale the out-of-sample data
X_out_of_sample_scaled = scaler_X.transform(X_out_of_sample)
X_out_of_sample_reshaped = np.reshape(X_out_of_sample_scaled, (X_out_of_sample_scaled.shape[0], 1, X_out_of_sample_scaled.shape[1]))

# Make predictions on out-of-sample data
predictions_out_of_sample_scaled = model.predict(X_out_of_sample_reshaped)
predictions_out_of_sample = scaler_y.inverse_transform(predictions_out_of_sample_scaled)

# Make predictions on test data
predictions_test_scaled = model.predict(X_test_reshaped)
predictions_test = scaler_y.inverse_transform(predictions_test_scaled)

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_test[:, 1], y_test[:, 0], label='True Values', color='green', marker='o')
ax.scatter(predictions_test[:, 1], predictions_test[:, 0], label='Predicted Values (Test Data)', color='blue', marker='x')
ax.scatter(predictions_out_of_sample[:, 1], predictions_out_of_sample[:, 0], label='Predicted Values (Out-of-Sample Data)', color='black', marker='x')
ax.set_title('Latitude and Longitude Prediction')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()
ax.grid(True)
st.pyplot(fig)

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


st.markdown("<h1 style='text-align: center;'>Internship Project</h1>", unsafe_allow_html=True)


df = pd.read_csv('combined_data.csv')
df['new_time'] = pd.to_datetime(df['time'], unit='s')
df['date1'] = df['new_time'].dt.date
df['time1'] = df['new_time'].dt.time
st.markdown("<h3 style='text-align: center;'>Top 5 Rows of Data</h3>", unsafe_allow_html=True)
st.write(df.head())


n = df['callsign'].nunique()
m = df['icao24'].nunique()
st.markdown(f"<h3 style='text-align: center;'>There are {n} Paths of {m} Planes</h3>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: center;'>Here you can Visualize the data</h3>", unsafe_allow_html=True)
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


st.markdown("<h3 style='text-align: center;'>Let say this is a common path for multiple flights</h3>", unsafe_allow_html=True)
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


st.markdown("<h3 style='text-align: center;'>Now we create a Boundary over these Paths Shows as Normal Flight Path</h3>", unsafe_allow_html=True)
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

from obspy import read_events, read_inventory
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
import numpy as np
def get_event_coordinates(station_lat, station_lon, distance_km, back_azimuth):
    geod = Geodesic.WGS84
    result = geod.Direct(station_lat, station_lon, back_azimuth, distance_km * 1000)
    return result['lon2'], result['lat2']

import pickle
with open('figures/realworldtw.npy','rb') as f:
    data = pickle.load(f)
(  axis,
real_x,
real_y,
pred_x,
pred_y,
real_x,
real_y,station_lat, station_lon,event_lon,
event_lat )= data

time=axis- 10

selected_index = np.concatenate([
np.where(time<0)[0][::50],
np.where(np.logical_and(time>0,time<3))[0][::10],
np.where(time>3)[0][::50],
]
)

x = pred_x[selected_index]
y = pred_y[selected_index]
T =   time[selected_index] 
distances     =  np.sqrt(x**2+y**2)
angles        = -np.rad2deg(np.arctan2(y, x))
real_distance = np.sqrt(real_x**2+real_y**2)
real_angle    = -np.rad2deg(np.arctan2(real_y, real_x))
real_lon, real_lat = get_event_coordinates(station_lat, station_lon, real_distance, real_angle)
pred_lonlats = [get_event_coordinates(station_lat, station_lon, d, baz) for d,baz in zip(distances, angles)]
predicted_lons=np.array([a for a,b in pred_lonlats])
predicted_lats=np.array([b for a,b in pred_lonlats])

distances     =  np.sqrt(x**2+y**2)
angles        = -np.rad2deg(np.arctan2(y, x))
real_distance = np.sqrt(real_x**2+real_y**2)
real_angle    = -np.rad2deg(np.arctan2(real_y, real_x))

real_lon, real_lat = get_event_coordinates(station_lat, station_lon, real_distance, real_angle)
pred_lonlats = [get_event_coordinates(station_lat, station_lon, d, baz) for d,baz in zip(distances, angles)]
predicted_lons=np.array([a for a,b in pred_lonlats])
predicted_lats=np.array([b for a,b in pred_lonlats])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())


# Add map features (e.g., coastlines, borders)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.add_feature(cfeature.LAND, color='lightgray')

# Plot the event location

ax.plot(event_lon, event_lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree())
ax.plot(station_lon, station_lat, marker='^', color='black', markersize=20, transform=ccrs.PlateCarree())
ax.scatter(real_lon, real_lat, marker='*', color='red', s=100, transform=ccrs.PlateCarree(), label='Predicted Quake')
station_buffer = 1  # Adjust this value to control the map extent
# map_extent = [event_lon - 5, event_lon + 5, event_lat - 2, event_lat + 2]
station_buffer
map_extent = [event_lon- 1.5, event_lon + 0.5,
              event_lat- station_buffer, event_lat + station_buffer]

# map_extent = [
#     min(sta.longitude for net in inventory for sta in net) - station_buffer,
#     max(sta.longitude for net in inventory for sta in net) + station_buffer,
#     min(sta.latitude for net in inventory for sta in net) - station_buffer,
#     max(sta.latitude for net in inventory for sta in net) + station_buffer
# ]
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ax.set_extent(map_extent, crs=ccrs.PlateCarree())
ooooi=0
offset_x=[-20, -20, -40, 20, -20, 20, 20, 20]
offset_y=[ 25,  25,  25, 25,  25, 25, 25, 25]
for i in range(len(predicted_lons) - 1):
    dx = predicted_lons[i+1] - predicted_lons[i]
    dy = predicted_lats[i+1] - predicted_lats[i]
    arrow_length = 0.9  # Adjust this value to control the arrow length
    ax.arrow(predicted_lons[i], predicted_lats[i], dx * arrow_length, dy * arrow_length,
              transform=ccrs.PlateCarree()._as_mpl_transform(ax),
              head_width=0.02, head_length=0.03, fc='blue', ec='blue', length_includes_head=True)
    time_now = T[i]
    annotation = f"T={time_now:0.1f}s"
    if time_now<0:continue
    if time_now>2:continue
    
    ax.annotate(annotation, (predicted_lons[i], predicted_lats[i]),
                xytext=(offset_x[ooooi], offset_y[ooooi]), textcoords='offset points',
                fontsize=10, color='black',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    ooooi +=1
ax.scatter(predicted_lons, predicted_lats, marker='.', color='green', s=100, transform=ccrs.PlateCarree(), label='Predicted Quake')

plt.title('Event and Station Location')
plt.xlabel('Longitude')

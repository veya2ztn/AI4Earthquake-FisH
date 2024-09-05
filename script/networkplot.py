import networkx as nx
import community
import matplotlib.pyplot as plt
import numpy as np
import os

ROOTDIR="/mnt/workspace/liufeng/project/01_DataPrepare/STEAD/data"

station_event_map = np.load(os.path.join(ROOTDIR,"station_event_map.npy"))

station_id = np.load(os.path.join(ROOTDIR,"station_id.npy"),allow_pickle=True)

station_to_station_count = np.load(os.path.join(ROOTDIR,"station_to_station_count.npy"))
station_to_station_count[np.diag_indices(
    station_to_station_count.shape[0])] = 0
edgepair = np.stack(np.where(station_to_station_count)).T

G = nx.Graph()
# num_nodes = len(station_id)
# nodes = range(num_nodes)
# G.add_nodes_from(nodes)
connections = edgepair
G.add_edges_from(connections)

# Create a graph object and add nodes and connections as described in the previous response

# Apply the Louvain algorithm to detect communities
partition = community.best_partition(G)

# Set up plot figure and axes
fig, ax = plt.subplots(figsize=(20, 20))

# Assign a unique color to each community
cmap = plt.get_cmap("viridis")
node_colors = [cmap(partition[node]) for node in G.nodes()]

# Draw the graph with node colors
pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=10, ax=ax)

# Save the plot as a PNG file
plt.savefig("graph.png", dpi=300)
# nx.draw(G, with_labels=False, node_size=10)
# plt.show()
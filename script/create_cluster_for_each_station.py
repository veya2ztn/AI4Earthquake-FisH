from train_single_station_via_llm_way import *
from dataset.TraceDataset import *
import json
from dataset.resource.resource_arguements import ResourceDiTing
from dataset.resource.load_resource import load_resource
import os
import copy
import scipy
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn import metrics
import sys
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

assert len(sys.argv)==3, "Usage: python create_cluster_for_each_station.py split_num split_idx"
split_num = int(sys.argv[1])
split_idx = int(sys.argv[2])
assert split_idx < split_num   
SAVEDIR = f"datasets/DiTing330km/SubStaID/Partition_for_{split_num}"
os.makedirs(SAVEDIR, exist_ok=True)
TARGETPATH= os.path.join(SAVEDIR, f"Partion_{split_idx}.json")
if os.path.exists(TARGETPATH):
    print(f"Already exists {TARGETPATH}")
    exit()

def find_the_best_eps(data):
    x_min, x_max = np.percentile(data['level_x'], [10,  90])
    y_min, y_max = np.percentile(data['level_y'], [10,  90])
    z_min, z_max = np.percentile(data['level_z'], [10,  90])
    fdata = data[(data['level_x'] < x_max) & (data['level_x'] > x_min) &
                (data['level_y'] < y_max) & (data['level_y'] > y_min) &
                (data['level_z'] < z_max) & (data['level_z'] > z_min)]
    X = np.stack([normalized(fdata.level_x), normalized(fdata.level_y), normalized(fdata.level_z)],-1)


    # db = DBSCAN(eps=.5, min_samples=10).fit(X)
    # labels = db.labels_
    # for trace_name, sub_label in zip(data['trace_name'], labels):
    #     if sub_label== -1:continue
    #     sub_sta_id = f"{selected_sta_id}-{sub_label}"
    #     sub_sta_id_map[trace_name] = sub_sta_id
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)

    eps_values = np.arange(0.05, 1.0, 0.05)  # Adjust this based on the domain and the scale of your data
    min_samples_values = [10]  # Adjust this based on prior knowledge of the data

    # Initialize best score for comparison
    best_score = -1
    best_params = {'eps': None, 'min_samples': None}

    # Grid search
    for eps in tqdm(eps_values, position=1, leave=False):
        for min_samples in min_samples_values:
            # Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = db.fit_predict(X)
            if len(set(clusters)) == 1 or len(set(clusters)) == len(X):continue
            #silhouette = metrics.silhouette_score(X, clusters)
            calinski_harabasz = metrics.calinski_harabasz_score(X, clusters)
            #davies_bouldin = metrics.davies_bouldin_score(X, clusters)
            #print(f"Silhouette Score: {silhouette} Calinski-Harabasz Index: {calinski_harabasz} Davies-Bouldin Index: {davies_bouldin}")
            score = calinski_harabasz
            # Save parameters with the best score
            #print(score)
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
    return best_params
# Rerousce = ResourceDiTing(resource_source='diting.group.full.good.hdf5')
# df, dtfl, index_map, normer = load_resource(Rerousce, only_metadata=False)
# filelist = [t for t in os.listdir('datasets/DiTing330km/WaveFormMean') if 'tracename' in t]
# name_and_wavemean_map = {}
# for filepath in filelist:
#     name_path = os.path.join("datasets/DiTing330km/WaveFormMean", filepath)
#     level_path  = os.path.join("datasets/DiTing330km/WaveFormMean", filepath.replace('tracename','mean_level'))
#     for name, level in zip(np.load(name_path), np.load(level_path)):
#         name_and_wavemean_map[name] = level

# df = df[df['trace_name'].isin(name_and_wavemean_map)]
# df['mean_level'] = [name_and_wavemean_map[trace_name] for trace_name in df['trace_name']]

# with open(f"datasets/DiTing330km/NameStaIDMap.json",'r') as f:key_staid_map = json.load(f)
# staid = [key_staid_map[str(trace_name)] for trace_name in df['trace_name']]
# df['sta_id'] = staid 
# del key_staid_map
# del name_and_wavemean_map

# station_tracelist = copy.deepcopy(df.groupby('sta_id')[['trace_name','mean_level']].apply(lambda x:[[t]+list(b) for t,b in zip(x.trace_name, x.mean_level.values)]))
# station_tracelist.to_csv("datasets/DiTing330km/NameMeanLevel.csv", index=None)
station_tracelist = pd.read_csv("datasets/DiTing330km/NameMeanLevel.csv")
print(len(station_tracelist))

indexes       = np.arange(len(station_tracelist))
all_partition = np.array_split(indexes, split_num)
now_partition = all_partition[split_idx]

def normalized(x):
#     std = x.std()
#     if std==0: std=1
#     return (x - x.mean())/std
#     scaler = StandardScaler()
#     return scaler.fit_transform(x.values.reshape(-1, 1))[:,0]
    robust_scaler = RobustScaler()
    return robust_scaler.fit_transform(x.values.reshape(-1, 1))[:,0]

sub_sta_id_map={}
for num in tqdm(now_partition, position=0, leave=True):
    selected_row = station_tracelist.iloc[num]
    selected_sta_id=selected_row['sta_id']
    data =  eval(selected_row['0'])
    data = pd.DataFrame(data, columns=['trace_name', 'level_x','level_y','level_z'])
    if len(data)==1:continue
    best_params = find_the_best_eps(data)
    X = np.stack([normalized(data.level_x), normalized(data.level_y), normalized(data.level_z)],-1)
    db = DBSCAN(**best_params).fit(X)
    #db = DBSCAN(eps=.05, min_samples=10).fit(X)
    labels = db.labels_
    for trace_name, sub_label in zip(data['trace_name'], labels):
        if sub_label== -1:continue
        sub_sta_id = f"{selected_sta_id}-{sub_label}"
        sub_sta_id_map[trace_name] = {'id':sub_sta_id}|best_params

#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     n_noise_ = list(labels).count(-1)

#     print('Estimated number of clusters: %d' % n_clusters_)
#     print('Estimated number of noise points: %d' % n_noise_)

#     # Plot the clusters

#     plt.scatter(X[labels != -1, 0], X[labels != -1, 1],s=0.5,alpha=0.5, c=labels[labels != -1], cmap='rainbow', label='Clusters')
#     #plt.scatter(X[labels == -1, 0], X[labels == -1, 1],s=0.5,alpha=0.5, c='black', label='Noise')
#     plt.title('DBSCAN Clustering')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend()
#     plt.show()
with open(TARGETPATH,'w') as f:
    json.dump(sub_sta_id_map, f)

# sub_sta_id_map ={}
# ROOTDIR="datasets/DiTing330km/SubStaID/Partition_for_50"
# for filename in os.listdir(ROOTDIR):
#     filepath = os.path.join(ROOTDIR, filename)
#     with open(filepath,'r') as f: sub_sta_id_map_this_partition = json.load(f)
#     sub_sta_id_map = sub_sta_id_map|sub_sta_id_map_this_partition

# import json
# with open(f"datasets/DiTing330km/SubStaID.eps0.05.key.json",'w') as f:
#     json.dump(sub_sta_id_map, f)
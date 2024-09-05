import h5py
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor
import tabulate
# data_path = "datasets/STEAD"
# csv_file = os.path.join(data_path, "stead.base.csv")
# df = pd.read_csv(csv_file)
# df = df[(df.trace_category == 'earthquake_local')]
# df = df[df['source_latitude'].notnull() & df['source_longitude'].notnull() & df['source_depth_km'].notnull() & df['source_magnitude'] > 0]

# clustring_flag = 'source_id'
# print(f"there are {len(df)} trace good trace record in the data")
# print(f"the clustring_flag is {clustring_flag}")
# grouped_df = df.groupby("source_id")
# print(f"there are {len(grouped_df)} events in the data")
# group_size = df.groupby('source_id').size()
# # Append the group size to each row according to the size
# df['group_size'] = df.groupby('source_id')['source_id'].transform('count')
# group_size_counts = df['group_size'].value_counts().sort_index()

# # Filter the counts for group sizes from 1 to 9
# filtered_counts = group_size_counts[group_size_counts.index.isin(range(1, 10))]

# # Convert the filtered counts to a pandas DataFrame
# group_size_table = pd.DataFrame(filtered_counts).reset_index()
# group_size_table.columns = ['group_size', 'count']
# print(tabulate.tabulate(group_size_table.transpose(),headers='', tablefmt='psql', showindex=True))

# for group_size in range(1,8):
#     newdf = df[df['group_size']==group_size]
#     event_names = np.array(newdf['trace_name'].values)
#     np.save(f"datasets/STEAD/LJSource/stead.event_names.L.{group_size}.new.grouped_by_source_id.npy",event_names)
# newdf = df[df['group_size']>=8]
# event_names = np.array(newdf['trace_name'].values)
# np.save(f"datasets/STEAD/LJSource/stead.event_names.L.b5.new.grouped_by_source_id.npy",event_names)

# print("save csv file ..........")
# df['labels'] = df["source_id"]
# output_file = "datasets/STEAD/stead.grouped_by_source_id.csv"
# df.to_csv(output_file, index=True)
# print("done")


# event_names = np.array(df['trace_name'].values)
# np.save(f"datasets/STEAD/stead.event_names.grouped_by_source_id.npy", event_names)
# exit()

# csv_file = "datasets/STEAD/stead.grouped.csv"
# df       = pd.read_csv(csv_file)
# df = df[(df.trace_category == 'earthquake_local')]
# df = df[df['source_latitude'].notnull() & df['source_longitude'].notnull() & df['source_depth_km'].notnull() & df['source_magnitude']>0]
# df = df[df['group_size'] > 1]
# event_names = np.array(df['trace_name'].values)
# np.save(f"datasets/STEAD/stead.event_names.grouped.npy",event_names)
# raise

f = h5py.File("datasets/STEAD/stead.hdf5", 'r')


def read_event_data(event_name_sub_list):
    if len(event_name_sub_list) == 0:
        return []
    #with h5py.File("datasets/STEAD/stead.hdf5", 'r') as f:
    g_event = np.zeros((len(event_name_sub_list), 6000, 3))
    for i, event_name in enumerate(event_name_sub_list):
        g_event[i] = np.array(f.get(f'data/{event_name}'))
    #g_event = [np.array(f.get(f'data/{event_name}')) for event_name in event_name_sub_list]
    return g_event

for group_size in range(1, 8):
    event_names = np.load(f"datasets/STEAD/LJSource/stead.event_names.L.{group_size}.new.grouped_by_source_id.npy",allow_pickle=True)
    print(len(event_names))
    num = len(event_names)//5000
    event_name_list = event_names[:5000*num]
    event_name_list = np.split(event_name_list,5000)
    event_name_list += [event_names[5000*num:]]
    with ProcessPoolExecutor(max_workers=64) as executor:
        picked_inputs_waveform = list(tqdm(executor.map(read_event_data, event_name_list), total=len(event_name_list)))
    picked_inputs_waveform = [t for t in picked_inputs_waveform if len(t)>0]
    waveform_data = np.concatenate(picked_inputs_waveform)
    np.save(f"datasets/STEAD/LJSource/stead.grouped.L.{group_size}.npy", waveform_data)
    #np.save(f"datasets/STEAD/LJSource/stead.grouped.index.npy", event_names)
    print("done")

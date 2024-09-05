import pandas as pd 
import os
from tqdm import tqdm
import multiprocessing as mp
import json

metadata = pd.read_csv("station.info.clean.csv")
ROOTPATH='datasets/STEAD/query.station.xmls/'
chunk_name_list = os.listdir(ROOTPATH)
path_list = mp.Manager().list([""]*len(metadata))
path_good = mp.Manager().list([False]*len(metadata))

def process_files(chunk_name):
    chunk_path = os.path.join(ROOTPATH, chunk_name)
    for xml_name in os.listdir(chunk_path):
        xml_path = os.path.join(chunk_path, xml_name)
        xml_index= int(xml_name.replace('.xml',""))
        path_list[xml_index]= xml_path
        if os.path.getsize(xml_path)>0:
            path_good[xml_index]= True

# Create a pool of processes
with mp.Pool(mp.cpu_count()) as pool:
    # Note: tqdm here is to show progress, it may not work perfectly with multiprocessing
    list(tqdm(pool.imap(process_files, chunk_name_list), total=len(chunk_name_list)))

with open("xmls_path_list.json",'w') as f:
    json.dump(list(path_list),f)

with open("xmls_path_good.json",'w') as f:
    json.dump(list(path_good),f)


# import pandas as pd 
# import os
# from tqdm.notebook import tqdm
# metadata = pd.read_csv("station.info.clean.csv")

# import json
# # path_list = [""]*len(metadata)
# # path_good = [False]*len(metadata)

# with open("xmls_path_list.json",'r') as f:path_list = json.load(f)
# with open("xmls_path_good.json",'r') as f:path_good = json.load(f)

# from tqdm.notebook import tqdm

# ROOTPATH='datasets/STEAD/query.station.xmls2/'

# index_and_path = []
# for chunk_name in tqdm(os.listdir(ROOTPATH)):
#     chunk_path = os.path.join(ROOTPATH, chunk_name)
#     for xml_name in os.listdir(chunk_path):
#         xml_path = os.path.join(chunk_path, xml_name)
#         xml_index= int(xml_name.replace('.xml',""))
#         if path_list[xml_index] != "":
#             print(f"{xml_index} duplicate")
#         path_list[xml_index]= xml_path
#         if os.path.getsize(xml_path)>0:
#             path_good[xml_index]= True

# metadata['inventory_path'] = path_list

# metadata['has_inventory']=path_good
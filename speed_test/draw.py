import json,re
pattern = r"SET:(.*?)=> (\d+\.\d+)"


records_3090 = {}
with open("speed_test.3090.txt.old",'r') as f:
    for line in f:

        matches = re.findall(pattern, line.strip())[0]
        setting = matches[0].strip()
        model, training_setting   = setting.split("|")
        training_setting = training_setting.split(".")
        amp_type = ".".join(training_setting[:2])
        num_GPU  = int(training_setting[2][1:])
        batch_size = int(training_setting[3][1:])
        cost = float(matches[1])
        if model not in records_3090:records_3090[model] = {}
        if amp_type not in records_3090[model]:records_3090[model][amp_type]={}
        if num_GPU not in records_3090[model][amp_type]:records_3090[model][amp_type][num_GPU]=[]
        
        records_3090[model][amp_type][num_GPU].append([batch_size,cost])

records_A10040 = {}
with open("speed_test.A10040.txt.old",'r') as f:
    for line in f:
        matches = re.findall(pattern, line.strip())[0]
        setting = matches[0].strip()
        model, training_setting   = setting.split("|")
        training_setting = training_setting.split(".")
        amp_type = ".".join(training_setting[:2])
        num_GPU  = int(training_setting[2][1:])
        batch_size = int(training_setting[3][1:])
        cost = float(matches[1])
        if model not in records_A10040:records_A10040[model] = {}
        if amp_type not in records_A10040[model]:records_A10040[model][amp_type]={}
        if num_GPU not in records_A10040[model][amp_type]:records_A10040[model][amp_type][num_GPU]=[]
        records_A10040[model][amp_type][num_GPU].append([batch_size,cost])
                        

records_A10080 = {}
with open("speed_test.A10080.txt.old",'r') as f:
    for line in f:
        matches = re.findall(pattern, line.strip())[0]
        setting = matches[0].strip()
        model, training_setting   = setting.split("|")
        training_setting = training_setting.split(".")
        amp_type = ".".join(training_setting[:2])
        num_GPU  = int(training_setting[2][1:])
        batch_size = int(training_setting[3][1:])
        cost = float(matches[1])
        if model not in records_A10080:records_A10080[model] = {}
        if amp_type not in records_A10080[model]:records_A10080[model][amp_type]={}
        if num_GPU not in records_A10080[model][amp_type]:records_A10080[model][amp_type][num_GPU]=[]
        records_A10080[model][amp_type][num_GPU].append([batch_size,cost])

records_3090_2x2 = {}
with open("speed_test.on3090.2x2.txt",'r') as f:
    for line in f:
        matches = re.findall(pattern, line.strip())[0]
        setting = matches[0].strip()
        model, training_setting   = setting.split("|")
        training_setting = training_setting.split(".")
        amp_type = ".".join(training_setting[:2])
        num_GPU  = int(training_setting[2][1:])
        batch_size = int(training_setting[3][1:])
        cost = float(matches[1])
        if model not in records_3090_2x2:records_3090_2x2[model] = {}
        if amp_type not in records_3090_2x2[model]:records_3090_2x2[model][amp_type]={}
        if num_GPU not in records_3090_2x2[model][amp_type]:records_3090_2x2[model][amp_type][num_GPU]=[]
        records_3090_2x2[model][amp_type][num_GPU].append([batch_size,cost])

records_A10040_2x2 = {}
with open("speed_test.onA10040G.2X2.txt",'r') as f:
    for line in f:
        matches = re.findall(pattern, line.strip())[0]
        setting = matches[0].strip()
        model, training_setting   = setting.split("|")
        training_setting = training_setting.split(".")
        amp_type = ".".join(training_setting[:2])
        num_GPU  = int(training_setting[2][1:])
        batch_size = int(training_setting[3][1:])
        cost = float(matches[1])
        if model not in records_A10040_2x2:records_A10040_2x2[model] = {}
        if amp_type not in records_A10040_2x2[model]:records_A10040_2x2[model][amp_type]={}
        if num_GPU not in records_A10040_2x2[model][amp_type]:records_A10040_2x2[model][amp_type][num_GPU]=[]
        records_A10040_2x2[model][amp_type][num_GPU].append([batch_size,cost])
        
import matplotlib.pyplot as plt
import scienceplots
model = 'Goldfish.40M_A.Sea'
with plt.style.context('no-latex','science'):
    fig, axes = plt.subplots(3, 1,dpi=400)
    record_3090 = records_3090_2x2[model]
    record_A10040=records_A10040_2x2[model]
    #record_A10080=records_A10080[model]
    color3090 = 'b'
    colorA10040 = 'r'
    #colorA10080 = 'g'
    Bmap = {1:1, 2:2, 4:3, 8:4, 16:5, 32:6, 64:7}
    for i,(amp_type, ax_row) in enumerate(zip(['DEEPSPEED.bf16','MULTI_GPU.bf16','MULTI_GPU.no'],axes)):
            j = 0
            num_GPU = 4
            ax = ax_row
            #for j,(num_GPU,ax) in enumerate(zip([4], ax_row)):
            ax.set_ylim([0,60])
        
            amp_type_real = amp_type.replace('MULTI_GPU','NO') if num_GPU == 1 else amp_type
            bar_width = 0.25
            for offset, (color, record,name) in enumerate(zip([color3090, colorA10040],
                                                               [record_3090,record_A10040],
                                                             ['3090','A10040G'])):
                if amp_type_real in record and num_GPU in record[amp_type_real]:

                    type_A = record[amp_type_real][num_GPU]
                    type_A = {k:v for k,v in type_A}
                    x_A, y_A = zip(*type_A.items())
                    x_A= [Bmap[t] for t in x_A]
                    r1 = [x + (offset - 1 )*bar_width for x in x_A]
                    ax.bar(r1, y_A, color=color, width=bar_width, edgecolor='grey', label=f'{name}')

            ax.set_title(f"{amp_type.replace('no','fp32')}.G{num_GPU}", fontweight='bold', fontsize=8)
            ax.legend()
            # Add xticks on the middle of the group bars
            if i==len(axes)-1:
                ax.set_xticks(list(Bmap.values()),list(Bmap.keys()))
                ax.set_xlabel('batch_size')
            else:
                ax.set_xticks([])
            ax.set_ylabel('Sample/s')

import matplotlib.pyplot as plt
import scienceplots
model = 'Goldfish.40M_A.Sea'
with plt.style.context('no-latex','science'):
    fig, axes = plt.subplots(3, 3,dpi=400)
    record_3090 = records_3090[model]
    record_A10040=records_A10040[model]
    record_A10080=records_A10080[model]
    color3090 = 'b'
    colorA10040 = 'r'
    colorA10080 = 'g'
    Bmap = {1:1, 2:2, 4:3, 8:4, 16:5, 32:6, 64:7}
    for i,(amp_type, ax_row) in enumerate(zip(['DEEPSPEED.bf16','MULTI_GPU.bf16','MULTI_GPU.no'],axes)):
        for j,(num_GPU,ax) in enumerate(zip([1,4,8], ax_row)):
            ax.set_ylim([0,60])
        
            amp_type_real = amp_type.replace('MULTI_GPU','NO') if num_GPU == 1 else amp_type
            bar_width = 0.25
            for offset, (color, record) in enumerate(zip([color3090, colorA10040,colorA10080 ],[record_3090,record_A10040,record_A10080] )):
                if amp_type_real in record and num_GPU in record[amp_type_real]:

                    type_A = record[amp_type_real][num_GPU]
                    type_A = {k:v for k,v in type_A}
                    x_A, y_A = zip(*type_A.items())
                    x_A= [Bmap[t] for t in x_A]
                    r1 = [x + (offset - 1 )*bar_width for x in x_A]
                    ax.bar(r1, y_A, color=color, width=bar_width, edgecolor='grey', label=f'{offset}')

            

            if i==0 and j==0:
                ax.scatter(1, 0, color=color3090, label='3090')
                ax.scatter(2, 0, color=colorA10040, label='A100|40G')
                ax.scatter(3, 0, color=colorA10080, label='A100|80G')
                ax.set_yticks([])
                ax.set_xlim([4,5])
                ax.legend()
            else:
                ax.set_title(f"{amp_type.replace('no','fp32')}.G{num_GPU}", fontweight='bold', fontsize=8)
            # Add xticks on the middle of the group bars
            if i==len(axes)-1:
                ax.set_xticks(list(Bmap.values()),list(Bmap.keys()))
                ax.set_xlabel('batch_size')
            else:
                ax.set_xticks([])
            if j==0 and i>0:
                ax.set_ylabel('Sample/s')
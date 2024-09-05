import os
import numpy as np
import argparse,tqdm,json,re
from mltool.loggingsystem import LoggingSystem
import json
import argparse
import re
from tbparse import SummaryReader
import os
import json
from mltool.loggingsystem import LoggingSystem
from config.utils import retrieve_dict
from project_arguements import to_dict
from project_arguements import get_args
import pandas as pd
import shutil
def remove_weight(trial_path):
    raise NotImplementedError


def assign_trail_job(trial_path,wandb_id=None, gpu=0):
    raise NotImplementedError

def run_finetune(path):
    raise NotImplementedError

def run_fourcast(ckpt_path,step = 4*24//6,force_fourcast=False,wandb_id=None,weight_chose=None):
    raise NotImplementedError

def remove_trail_path(trial_path):
    if "metric_report.json" not in os.listdir(trial_path):
        if "pytorch_model.bin" not in os.listdir(trial_path):
            shutil.rmtree(trial_path)
            return
    if "checkpoints" not in os.listdir(trial_path):
        if "pytorch_model.bin" not in os.listdir(trial_path):
            shutil.rmtree(trial_path)
    else:
        if len(os.listdir(os.path.join(trial_path, "checkpoints"))) == 0:
            shutil.rmtree(trial_path)
        else:
            pass
            # checkpoint_dir  = os.path.join(trial_path, "checkpoints")
            # checkpoint_name = os.listdir(checkpoint_dir)[0]
            # checkpoint_idx  = int(checkpoint_name.replace('checkpoint_',''))
            # checkpoint_dir  = os.path.join(checkpoint_dir, checkpoint_name)
            # if os.path.exists(f"{checkpoint_dir}/clean_checkpoints_via_finish_train"):
            #     return
            # # if "10M" in checkpoint_dir:
            # os.system(f"rm -rf {checkpoint_dir}/*")
            # os.system(f"touch {checkpoint_dir}/clean_checkpoints_via_finish_train")
            # if checkpoint_idx < 90:
            #     return
            #     os.system(f"rm -rf {trial_path}")
            #print(checkpoint_dir)
            #print(f"[CK:{checkpoint_idx}]: rm -r {trial_path}")
    # if 'trace' in os.listdir(trial_path):
    #     number_of_images_list = []
    #     for porperty_type in os.listdir(os.path.join(trial_path,'trace')):
    #         number_of_images = len(os.listdir(os.path.join(trial_path,'trace',porperty_type)))
    #         number_of_images_list.append(number_of_images)
    #     if max(number_of_images_list)<10:
    #         os.system(f"rm -r {trial_path}")
    # else:
    #     pass
    #     #os.system(f"rm -r {trial_path}")
    #raise NotImplementedError

def remove_checkpoint_under_best_path(trial_path):
    if "best" not in os.listdir(trial_path):return
    best_path = os.path.join(trial_path, 'best')
    for filename in os.listdir(best_path):
        filepath = os.path.join(best_path, filename)
        checkpointpath = os.path.join(filepath, 'checkpoint')
        if os.path.islink(checkpointpath):
            ## remove link
            os.unlink(checkpointpath)
        elif os.path.exists(checkpointpath):
            shutil.rmtree(checkpointpath)
    
def remove_visulization_path(trial_path):
    if "visualize" not in os.listdir(trial_path):return
    visualize_path = os.path.join(trial_path, 'visualize')
    if os.path.exists(visualize_path):
        shutil.rmtree(visualize_path)


def remove_evaluation(trial_path):
    out, should_result = test_evaluator.evaluate(model, logsys)
    compare_the_distribution(args, out, should_result,
                             root_path=args.SAVE_PATH, epoch=epoch)
    out = test_evaluator.compute_once_metric(out,should_result)
    print(out)
    raise NotImplementedError

def create_logsys(args,save_config=True):
    local_rank = 0
    SAVE_PATH  = args.output_dir
    recorder_list = ['wandb_runtime']
    use_wandb  = 'wandb_runtime' if 'wandb' in recorder_list else False
    logsys   = LoggingSystem(local_rank==0,SAVE_PATH,seed=args.task.seed,
                             use_wandb=use_wandb, recorder_list=recorder_list, flag='train',
                             disable_progress_bar=False)
    hparam_dict={}
    metric_dict={}
    dirname          = SAVE_PATH
    dirname,name     = os.path.split(dirname)
    _ = logsys.create_recorder(hparam_dict={},metric_dict={},
                                args    = retrieve_dict(to_dict(args)),
                                project = f"{args.dataset_name}-{args.model.Predictor.prediction_type}",
                                entity  = "szztn951357",
                                group   = args.model_name,
                                job_type= None,
                                name    = name
                               ) 
   
    #if local_rank == 0:print_namespace_tree(args)
    return logsys

def upload_trial_to_wandb(trial_path):
    import wandb
    from pathlib import Path
    
    events_pathlist = list(Path(trial_path).glob('*/events*'))
    config_pathlist = list(Path(trial_path).glob('train_config.json'))
    if len(config_pathlist)==0:
        print(f"==>no config at {trial_path},pass")
        return
    if len(events_pathlist)==0:
        print(f"==>no events at {trial_path},pass")
        return
    outdf = []
    for log_dir in events_pathlist:
        log_dir = str(log_dir)
        print(log_dir)
        reader = SummaryReader(log_dir)
        df = reader.scalars

        if len(df) < 1: 
            print(f"==>no scalars at,pass")
            continue
        print("===>start parsing tensorboard..............")
        outdf.append(df)
    #assert len(outdf)==1
    #df = outdf[0]
    df = pd.concat(outdf)
    args = get_args(config_pathlist[0],args=['--task','train',
                                             '--Resource','STEAD',
                                             #'--NoiseGenerate', 
                                             #'pickalong_receive'
                                             ]) # <-- should mannul set this
    args.output_dir = trial_path
    logsys = create_logsys(args, save_config=False)
    all_pool={}
    for step, tag, val in df.values:
        if step not in all_pool:all_pool[step]={}
        all_pool[step][tag]  =val 

    for step, record in all_pool.items():
        record['step'] = step
        wandb.log(record,step =step)

    logsys.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('parse tf.event file to wandb', add_help=False)
    parser.add_argument('--paths',type=str,default="")
    parser.add_argument('--moded',type=str,default="dryrun")
    parser.add_argument('--level', default=1, type=int)
    parser.add_argument('--divide', default=1, type=int)
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--fourcast_step', default=4*24//6, type=int)
    parser.add_argument('--path_list_file',default="",type=str)
    parser.add_argument('--force_fourcast',default=0,type=int)
    parser.add_argument('--weight_chose',default=None,type=str)
    args = parser.parse_known_args()[0]

    if args.paths != "":
        level = args.level
        root_path = args.paths
        now_path = [root_path]
        while level>0:
            new_path = []
            for root_path in now_path:
                if os.path.isfile(root_path):continue
                if len(os.listdir(root_path))==0:
                    os.system(f"rm -r {root_path}")
                    continue
                for sub_name in os.listdir(root_path):
                    sub_path =  os.path.join(root_path,sub_name)
                    if os.path.isfile(sub_path):continue
                    new_path.append(sub_path)
            now_path = new_path
            level -= 1
    
    now_path = [ p for p in now_path]
    now_path_pool = None
    if 'json' in args.path_list_file:
        with open(args.path_list_file, 'r') as f:
            path_list_file = json.load(f)
        if isinstance(path_list_file,dict):
            now_path_pool = path_list_file
            now_path = list(now_path_pool.keys())
        else:
            now_path = path_list_file

    print(f"we detect {len(now_path)} trail path;")
    print(f"from {now_path[0]} to \n  {now_path[-1]}")
    total_lenght = len(now_path)
    length = int(np.ceil(1.0*total_lenght/args.divide))
    s    = int(args.part)
    now_path = now_path[s*length:(s+1)*length]
    print(f"we process:\n  from  from {now_path[0]}\n to  {now_path[-1]}")
    

    if args.moded == 'dryrun':exit()
    for trail_path in tqdm.tqdm(now_path):
        trail_path = trail_path.rstrip("/")
        if len(os.listdir(trail_path))==0:
            os.system(f"rm -r {trail_path}")
            continue
        #os.system(f"sensesync sync s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ {trail_path}/")
        #os.system(f"sensesync sync {trail_path}/ s3://QNL1575AXM6DF9QUZDA9:BiulXCfnNpIx6tl6P14I8W5QDw6NSU3yqSWVdbkH@FourCastNet.10.140.2.204:80/{trail_path}/ ")
        #os.system(f"aws s3 --endpoint-url=http://10.140.2.204:80 --profile zhangtianning sync s3://FourCastNet/{trail_path}/ {trail_path}/")
        # print(trail_path)
        # print(os.listdir(trail_path))
        if   args.moded == 'fourcast':run_fourcast(trail_path,step=args.fourcast_step,force_fourcast=args.force_fourcast,weight_chose=args.weight_chose)
        elif args.moded == 'tb2wandb':upload_trial_to_wandb(trail_path)
        elif args.moded == 'cleantmp':remove_trail_path(trail_path)
        elif args.moded == 'cleanvis':remove_visulization_path(trail_path)
        elif args.moded == 'cleanwgt':remove_weight(trail_path)
        elif args.moded == 'cleancpt':remove_checkpoint_under_best_path(trail_path)
        elif args.moded == 'createtb':create_fourcast_table(trail_path,force_fourcast=args.force_fourcast)
        elif args.moded == 'snap_nodal':run_snap_nodal(trail_path,step=args.fourcast_step,force_fourcast=args.force_fourcast,weight_chose=args.weight_chose)
        elif args.moded == 'createtb_nodalsnap':create_nodalsnap_table(trail_path)
        elif args.moded == 'createmultitb':create_multi_fourcast_table(trail_path,force=args.force_fourcast)
        else:
            raise NotImplementedError

import os

def think_up_a_unique_name_for_inference(args):#args:ProjectConfig
    assert args.task.train_or_infer in ['infer','infer_plot'], "train_or_infer must be infer or infer_plot, but got {}".format(args.task.train_or_infer)
    name = f"{args.task.sampling_strategy.valid_sampling_strategy.strategy_name}.w{args.task.sampling_strategy.valid_sampling_strategy.early_warning}.l{args.DataLoader.Dataset.max_length}"
    if hasattr(args.DataLoader.Dataset, 'component_intervel_length'):
        name = name + f".c{args.DataLoader.Dataset.component_intervel_length}"
    if hasattr(args.DataLoader.Dataset, 'padding_rule'):
        name = name + f".Pad_{args.DataLoader.Dataset.padding_rule}"
    return name
def get_local_rank(args=None):
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    return local_rank
def get_rank(args=None):
    local_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    return local_rank
def print0(*args,**kwargs):
    if get_rank()==0:print(*args,**kwargs)

def printg(*args,**kwargs):
    print(f"GPU:[{get_local_rank()}]",*args,**kwargs)

def smart_load_weight(model, weight, strict=True, shape_strict=True):
    
    model_dict = model.state_dict()
    has_not_used_key = False
    has_missing_key = False
    if not strict:
        for key in weight.keys():
            if key not in model_dict:
                print0(f"====> key: {key} are not used in this model, and we will skip")
                has_not_used_key = True
                continue
        if not has_not_used_key:print0("All keys in pretrained weight are used in this model")
        for key in model_dict.keys():
            if key not in weight:
                print0(f"====> key: {key} missing, please check. So far, we pass and random init it")
                has_missing_key = True
                continue
        if not has_missing_key:print0("All keys in this model are in pretrained weight")
            
    if shape_strict:
        model.load_state_dict(weight,strict=strict) 
    else:
        assert not strict, "shape_strict=False and strict=True is not allowed"
        for key in model_dict.keys():
            if key not in weight:
                #print0(f"key: {key} missing, please check. So far, we pass and random init it")
                continue
            if model_dict[key].shape != weight[key].shape:
                #print0(f"shape mismatching: {key} {model_dict[key].shape} {weight[key].shape}")
                continue
            model_dict[key] = weight[key]  
        
        
        model.load_state_dict(model_dict,strict=False)

def is_ancestor(ancestor, descendent):
    ancestor = os.path.abspath(ancestor)
    descendent = os.path.abspath(descendent)
    return descendent.startswith(ancestor)
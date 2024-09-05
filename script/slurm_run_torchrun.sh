#!/bin/bash
#SBATCH -o log/TEST2.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/TEST2.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH --partition=AI4Phys
#SBATCH --job-name=XYZ
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2

GPUS_PER_NODE=2
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))



function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=1 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

echo "WORLD_SIZE:" $WORLD_SIZE "NODE_RANK:" $NODE_RANK "DEVICES:" $CUDA_VISIBLE_DEVICES
echo "MASTER_ADDR:" $MASTER_ADDR "MASTER_PORT:" $MASTER_PORT "SLURM_PROCID:" $SLURM_PROCID
echo "NNODES": $NNODES

export CMD="train_single_station_via_llm_way.py --model_type RetNetDecay_ReLU --epochs 300 --warning_window -4 --downstream_task mse_xyS --lr=0.0005 \
--seed=21 --batch_size 2 --num_hidden_layers 4 --hidden_size 512 --num_heads 16 --max_length 6000 --sched_cycle_nums 1 \
--dataset_name stead.trace.BDLEELSSO_sample6000 --wave_channel 3 --status_type N0P1S2 --warmup_epochs 10 --use_confidence status --debug"

export ACCELERATE_USE_DEEPSPEED=true #<----use this enable accelerate deepspeed
export ACCELERATE_DEEPSPEED_CONFIG_FILE=config/deepspeed/deepspeed_config_s1.json #<----use this enable accelerate deepspeed

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "
echo $LAUNCHER
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'

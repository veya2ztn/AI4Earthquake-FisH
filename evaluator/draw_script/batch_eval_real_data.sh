REALDATAFLAG=$1
export CUDA_VISIBLE_DEVICES=0;nohup python realdata_ploting.py checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/STEAD.EarlyWarning $REALDATAFLAG > log/convert/thread.0.log&
export CUDA_VISIBLE_DEVICES=1;nohup python realdata_ploting.py checkpoints/stead.trace.BDLEELSSO.hdf5/Goldfish.40M_A.Sea/02_16_23_23-seed_21-eb18 $REALDATAFLAG > log/convert/thread.1.log&
export CUDA_VISIBLE_DEVICES=2;nohup python realdata_ploting.py checkpoints/stead.trace.full.extend.hdf5/Goldfish.40M_A.Sea/02_06_13_07-seed_21-7567 $REALDATAFLAG > log/convert/thread.2.log&
export CUDA_VISIBLE_DEVICES=3;nohup python realdata_ploting.py checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/12_24_22_23_19-seed_21 $REALDATAFLAG > log/convert/thread.3.log&
export CUDA_VISIBLE_DEVICES=4;nohup python realdata_ploting.py checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/02_07_14_12-seed_21-892c $REALDATAFLAG > log/convert/thread.4.log&
export CUDA_VISIBLE_DEVICES=5;nohup python realdata_ploting.py checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/02_04_20_30-seed_21-a374 $REALDATAFLAG > log/convert/thread.5.log&
# nohup CUDA_VISIBLE_DEVICES=6;python realdata_ploting.py checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/STEAD.EarlyWarning $REALDATAFLAG > log/convert/thread.6.log&
# nohup CUDA_VISIBLE_DEVICES=7;python realdata_ploting.py checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/STEAD.EarlyWarning $REALDATAFLAG > log/convert/thread.7.log&
CPU_NUM=100 # Automatically get the number of CPUs
for ((CPU=1; CPU<CPU_NUM; CPU++));
do
    
    nohup python compute_stft_feature.py $CPU_NUM $CPU > log/convert/thread.$CPU.log&
done 
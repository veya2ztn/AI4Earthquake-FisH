CPU_NUM=200 # Automatically get the number of CPUs
for ((CPU=0; CPU<CPU_NUM; CPU++));
do
    #taskset -c $CPU 
    nohup python convert_wave_to_accelerate.py $CPU $CPU_NUM > log/convert/thread.$CPU.log&
done 

CPU_NUM=100 # Automatically get the number of CPUs
for ((CPU=0; CPU<CPU_NUM; CPU++));
do
    nohup python run.py $CPU_NUM $CPU > log/convert/thread.$CPU.log&
done 
#
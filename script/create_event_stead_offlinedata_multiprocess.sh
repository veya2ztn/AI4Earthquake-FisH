for CPU in {0..100};
do
    nohup taskset -c $CPU python script/create_event_stead_offlinedata_multiprocess.py $CPU > log/convert/thread.$CPU.log&
done 
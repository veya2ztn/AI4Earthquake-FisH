for ORDER in {0..50};
do
    nohup python create_cluster_for_each_station.py 50 $ORDER > log/convert/thread.$ORDER.log &
done

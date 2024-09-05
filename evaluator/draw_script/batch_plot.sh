for THEPATH in \
checkpoints/stead.trace.BDLEELSSO/Goldfish.40M_A.Sea/GoldfishS40MA_status_6000/visualize/DEV/ahead_L_to_the_sequence.w3000.l27000.c9000_data;
do
    for EARLYWARNING in 200;
    do
        if [[ -e "$THEPATH" ]];then
            python train_single_station_via_llm_way.py -c $THEPATH/infer_config.json --task infer_plot --Resource STEAD --plot_data_dir $THEPATH --upload_to_wandb False --Dataset ConCatDataset
        else
            echo "Target path $THEPATH does not exist, skipping..."
        fi
    done
done
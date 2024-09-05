for MODEL in GoldfishS40M GoldfishS10M;
do
    for batch_size in 16 32 64;
    do
        for setup_commend in "python" \
        "accelerate-launch --config_file config/accelerate/bf16_cards8_ds1.yaml" \
        "accelerate-launch --config_file config/accelerate/bf16_cards8.yaml" \
        "accelerate-launch --config_file config/accelerate/fp32_cards8.yaml" \
        "accelerate-launch --config_file config/accelerate/bf16_cards4.yaml" \
        "accelerate-launch --config_file config/accelerate/fp32_cards4.yaml" \
        "accelerate-launch --config_file config/accelerate/bf16_cards4_ds1.yaml" \
        "accelerate-launch --config_file config/accelerate/bf16_single.yaml" \
        "accelerate-launch --config_file config/accelerate/fp32_single.yaml";
        
        
        do 
            $setup_commend train_single_station_via_llm_way.py --task train --model $MODEL --epochs 1 \
            --downstream_task status --lr=0.0001 --seed=21 --batch_size $batch_size --max_length 6000 \
            --num_workers 8 --data_parallel_dispatch --resource_source stead.trace.BDLEELSSO \
            --debug --time_test
        done 
    done
done

exit

python train_single_station_via_llm_way.py --task train --model GoldfishS40M --epochs 1 \
            --downstream_task status --lr=0.0001 --seed=21 --batch_size 4 --max_length 6000 \
            --num_workers 8 --data_parallel_dispatch --resource_source stead.trace.BDLEELSSO \
            --debug --time_test
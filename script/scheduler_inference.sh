for THEPATH in \
checkpoints/stead.trace.full.extend.addRevTypeNoise/Goldfish.40M_A.Sea/04_13_22_41-seed_21-284c;
do
    for EARLYWARNING in 200;
    do
        if [[ -e "$THEPATH" ]];then

                #nohup python train_single_station_via_llm_way.py -c $THEPATH/infer_config.json --task infer_plot --plot_data_dir $THEPATH --upload_to_wandb > log/upload_to_wandb.log&
                
                ## findP ######
                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --model GoldfishS40M --preload_weight $THEPATH/mse_xymSfindPfindS_instance_unit.merge.weight.bin \
                # --Resource Instance --task recurrent_infer --downstream_task mse_xymSPS_Instance_unit \
                # --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --valid_sampling_strategy.early_warning 200 --normlize_at_downstream \
                # --batch_size 16 --max_length 12000 --recurrent_chunk_size 1500 --preload_state "" --upload_to_wandb True \
                # --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --load_weight_partial False --load_weight_ignore_shape False \
                # --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False --slide_feature_window_size 1500 --slide_stride_in_training 30 \

                ### findP in long sequence ####

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --model GoldfishS40M --preload_weight $THEPATH/pytorch_model.bin \
                # --Resource STEAD --task recurrent_infer \
                # --valid_sampling_strategy.strategy_name early_warning_before_p \
                # --valid_sampling_strategy.early_warning 6000 \
                # --batch_size 4 --max_length 12000 --recurrent_chunk_size 6000 --preload_state "" --upload_to_wandb False \
                # --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --load_weight_partial False --load_weight_ignore_shape False \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 1 \


                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --model GoldfishS40M --preload_weight $THEPATH/pytorch_model.bin \
                # --Resource STEAD --task recurrent_infer --resource_source stead.trace.BDLEELSSO.hdf5 \
                # --valid_sampling_strategy.strategy_name early_warning_before_p \
                # --valid_sampling_strategy.early_warning $EARLYWARNING \
                # --batch_size 16 --max_length 9000 --recurrent_chunk_size 3000 --preload_state "" --upload_to_wandb False \
                # --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --load_weight_partial False --load_weight_ignore_shape False \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False \
                # #--slide_feature_window_size 1000 --slide_stride_in_training 100 --slide_stride_in_training 10 \
                # # --padding_rule none --NoiseGenerate pickalong_receive --noise_config_tracemapping_path datasets/STEAD/name2receivetype.json --noise_config_noise_namepool datasets/STEAD/stead.noise.csv
                # # --padding_rule noise --NoiseGenerate pickalong_receive --noise_config_tracemapping_path datasets/STEAD/name2receivetype.json --noise_config_noise_namepool datasets/STEAD/stead.noise.csv

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
                # --Resource DiTing --task recurrent_infer --valid_sampling_strategy.strategy_name early_warning_before_p \
                # --valid_sampling_strategy.early_warning $EARLYWARNING \
                # --batch_size 32 --max_length 4512 --recurrent_chunk_size 4512 --preload_state "" --upload_to_wandb True \
                # --freeze_embedder "" --freeze_backbone "" --freeze_downstream ""  \
                # --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False #--resource_source diting.group.good.hdf5 #--resource_source ditinggroup.they.fast.cross.eval.hdf5

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
                # --Resource DiTing --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --valid_sampling_strategy.early_warning 3000 --Dataset ConCatDataset --component_concat_file datasets/DiTing330km/ditinggroup.full.subcluster.valid.cat2series.list.npy \
                # --batch_size 4 --max_length 27000 --recurrent_chunk_size 9000 --preload_state "" --upload_to_wandb False \
                # --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" --component_intervel_length 9000 \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False \
                # --slide_stride_in_training  100 --resource_source diting.group.full.good.XS.hdf5  --downstream_task mse_SL4 --load_weight_partial --dataset_version beta #\--resource_source ditinggroup.they.fast.cross.eval.hdf5

                accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
                --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name early_warning_before_p  \
                --valid_sampling_strategy.early_warning 500  \
                --batch_size 32 --max_length 12000 --recurrent_chunk_size 3000 --preload_state "" --upload_to_wandb True \
                --freeze_embedder "" --freeze_backbone "" --freeze_downstream ""  \
                --clean_up_plotdata True --use_resource_buffer False  --NoiseGenerate pickalong_receive


                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6  \
                # --max_length 27000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 1500 --recurrent_start_size 1500 --component_intervel_length 12000 \
                # --preload_state "" --upload_to_wandb False --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
                # --resource_source stead.trace.BDLEELSSO.hdf5 --monitor_retention True --retention_mode kv_first

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6  \
                # --max_length 18000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 1500 --recurrent_start_size 1500 --component_intervel_length 3000 \
                # --preload_state "" --upload_to_wandb False --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
                # --resource_source stead.trace.BDLEELSSO.hdf5 --monitor_retention True --retention_mode kv_first --use_zero_padding_in_concat

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6  \
                # --max_length 21000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 1500 --recurrent_start_size 1500 --component_intervel_length 6000 \
                # --preload_state "" --upload_to_wandb False --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
                # --resource_source stead.trace.BDLEELSSO.hdf5 --monitor_retention True --retention_mode kv_first --use_zero_padding_in_concat

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6  \
                # --max_length 24000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 1500 --recurrent_start_size 1500 --component_intervel_length 9000 \
                # --preload_state "" --upload_to_wandb False --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
                # --resource_source stead.trace.BDLEELSSO.hdf5 --monitor_retention True --retention_mode kv_first --use_zero_padding_in_concat

                
                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer \
                # --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6 \
                # --max_length 24000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 1500 --recurrent_start_size 1500 \
                # --component_intervel_length 9000 --preload_state "" --upload_to_wandb False --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 --model GoldfishS40M \
                # --resource_source stead.trace.BDLEELSSO.hdf5 \
                # --padding_rule noise --NoiseGenerate pickalong_receive \
                # --noise_config_tracemapping_path datasets/STEAD/name2receivetype.json \
                # --noise_config_noise_namepool datasets/STEAD/stead.noise.csv

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py \
                # -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin --Resource STEAD --task recurrent_infer \
                # --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence \
                # --Dataset ConCatDataset --component_concat_file datasets/STEAD/BDLEELSSO/stead.valid.cat2series.list.npy --batch_size 6 \
                # --max_length 24000 --max_length 24000 --valid_sampling_strategy.early_warning 3000 --recurrent_chunk_size 3000 \
                # --recurrent_start_size 3000 --component_intervel_length 9000 --preload_state --upload_to_wandb False \
                # --freeze_embedder --freeze_backbone --freeze_downstream --status_type N0P1S2 --use_confidence whole_sequence \
                # --clean_up_plotdata False --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 \
                # --model GoldfishS40M --resource_source stead.trace.BDLEELSSO.hdf5 --padding_rule zero \
                # --padding_rule noise 
                # --NoiseGenerate nonoise --noise_config_tracemapping_path --noise_config_noise_namepool \
                # --NoiseGenerate pickalong_receive --noise_config_tracemapping_path datasets/STEAD/name2receivetype.json \
                # --noise_config_noise_namepool datasets/STEAD/stead.noise.csv

                # accelerate-launch --main_process_port 12542 --config_file config/accelerate/bf16_cards8.yaml train_single_station_via_llm_way.py -c $THEPATH/train_config.json --preload_weight $THEPATH/pytorch_model.bin \
                # --Resource Instance --task recurrent_infer --valid_sampling_strategy.strategy_name ahead_L_to_the_sequence --Dataset ConCatDataset \
                # --component_concat_file datasets/INSTANCE/instance.valid.cat2series.list.npy --batch_size 6  --max_length 24000 --valid_sampling_strategy.early_warning 3000 \
                # --recurrent_chunk_size 1500 --recurrent_start_size 1500 --component_intervel_length 9000 --preload_state "" --upload_to_wandb True --freeze_embedder "" --freeze_backbone "" --freeze_downstream "" \
                # --clean_up_plotdata True --use_resource_buffer False --loader_all_data_in_memory_once False --slide_stride_in_training 100 \
                # --model GoldfishS40M  --padding_rule zero
                
            #fi
        else
            echo "Target path $THEPATH does not exist, skipping..."
        fi
    done
done
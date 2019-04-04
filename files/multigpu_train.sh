python multigpu_train.py \
	--gpu_list=3 \
	--input_size=512 \
	--batch_size_per_gpu=12 \
	--checkpoint_path=./checkpoints/ \
	--text_scale=512 \
	--training_data_list=/Dataset/MLT2019/trainMLT_ch_en.txt \
	--geometry=RBOX \
	--learning_rate=0.0002 \
	--num_readers=24 \
	# --pretrained_model_path=/tmp/resnet_v1_50.ckpt 	# Here we don't adopt a pre-trained model. If you want to use the pre-trained model, you need download it yourself (Please refer to https://github.com/argman/EAST)

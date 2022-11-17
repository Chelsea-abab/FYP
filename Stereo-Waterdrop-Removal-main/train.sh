cd ./FYP/Stereo-Waterdrop-Removal-main
CUDA_VISIBLE_DEVICES=4 nohup python train.py --save_name first_change 1>>./logs/first_change.log

tensorboard --logdir=./FYP/Stereo-Waterdrop-Removal-main/result/first_trail/logs --port 8125

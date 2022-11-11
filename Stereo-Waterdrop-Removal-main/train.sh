cd ./FYP/Stereo-Waterdrop-Removal-main
CUDA_VISIBLE_DEVICES=2 nohup python train.py --save_name first_trail 1>>./logs/first_trail.log

tensorboard --logdir=./FYP/Stereo-Waterdrop-Removal-main/result/first_trail/logs --port 8125

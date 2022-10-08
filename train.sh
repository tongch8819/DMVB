date_str=$(date +"%Y-%m-%d")
python train.py \
    --agent bmv \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./exp/cheetah_run_$date_str \
    --seed 1 \
    --crop_size 68 \
    1>exp/train.log 2>&1 &
echo $! >exp/pid.txt

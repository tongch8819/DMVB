python train.py \
    --domain_name cheetah \
    --task_name run \
    --encoder_type pixel \
    --decoder_type identity \
    --action_repeat 4 \
    --save_video \
    --save_tb \
    --work_dir ./log \
    --seed 1 \
    --agent bmv \
    1>exp/train.log 2>&1 &
echo $! >exp/pid.txt

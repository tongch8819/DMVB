python BSIBO/bisim.py \
    --num_of_rollout_steps 100000 \
    --num_of_training_steps 10000 >exp/bisim.log 1>&2
echo -e "Finished Bisim training at: $(date) \n" >>exp/bisim.log

CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type rssm --work_dir ./exp \
    --action_repeat 8 --num_eval_episodes 8 \
    --pre_transform_image_size 100 --image_size 84 --kl_balance \
    --agent BSIBO_sac --frame_stack 1 --encoder_feature_dim 1024 --save_model \
    --seed 0 --critic_lr 1e-5 --actor_lr 1e-5 --eval_freq 10000 --batch_size 8 --num_train_steps 890000 --save_model \
    --save_tb \
    1>exp/train.log

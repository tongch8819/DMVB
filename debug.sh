# python train.py \
# --replay_buffer_capacity 100 \
# --init_steps 10 \
# --num_train_steps 1000 \
# --eval_freq 100


# CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0 python train_new.py \
CUDA_VISIBLE_DEVICES=0 python trainv2.py \
--domain_name finger \
--task_name spin \
--encoder_type rssm \
--work_dir ./exp \
--action_repeat 8 \
--num_eval_episodes 8 \
--pre_transform_image_size 100 \
--image_size 84 \
--kl_balance \
--agent BSIBO_sac \
--frame_stack 1 \
--encoder_feature_dim 1024 \
--save_model  \
--seed 0 \
--critic_lr 1e-5 \
--actor_lr 1e-5 \
--eval_freq 100 \
--batch_size 8 \
--num_train_steps 1000 \
--init_steps 10 \
--replay_buffer_capacity 100 \
--init_steps 10 \
--mib_seq_len 5 

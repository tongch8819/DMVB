# BSIBO: BiSimulation Information BOttleneck based reinforcement learning


Debug notes for DRIBO
1. --image_size must be the same with DRIBO repo;
2. mib_seq_len should be less than 10 for evaluation sample;
3. replay buffer capacity must be small for debug case since large buffer need 11GB memory and desktop does not have that much memory for computation.

DBC
1. reward loss for transition model optimization; maybe replace it with dynamic programming technique of bisimulation
2. actor has a component called encoder
3. tied weights between encoders in actor and critic


Technical Selection
1. actor, critic, encoder are three to-be-optimized neural network
2. obs -- encoder --> representation -- actor --> action


todo
Debug architecture
Carpole
obs_space    Box(0, 1, (3,100,100), uint8)
action_space (1,)
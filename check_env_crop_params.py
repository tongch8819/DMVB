from train import parse_args
from BSIBO.utils import utils
from BSIBO.utils.video import VideoRecorder
import dmc2gym
import matplotlib.pyplot as plt
import numpy as np

def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        # resource_files=args.resource_files,
        # img_source=args.img_source,
        # total_frames=args.total_frames,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.image_size,
        width=args.image_size,
        frame_skip=args.action_repeat
    )
    env.seed(args.seed)
    # obs = env.observation_space.sample()
    # env.reset()
    # env.render()
    # plt.imshow(np.transpose(obs, (1,2,0)))
    # plt.savefig('tmp.png')
# 
    # print('\n\n#######################')
    # print(args.domain_name, args.task_name)

    video = VideoRecorder('./video')

    obs = env.reset()
    video.init(True)
    epi_rew = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            break
        epi_rew += reward
        print(i, epi_rew)
        video.record(env)
    video.save('tmp.mp4')

if __name__ == "__main__":
    main()
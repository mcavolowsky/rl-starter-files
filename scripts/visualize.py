import argparse, time, sys
import numpy
import torch

import utils

import gym
import gym_minigrid

from training import TaskDescriptor
import GoalRL
from gym_minigrid.envs.goaldescriptor import *

def main():
    # Parse arguments

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    parser.add_argument("--env", required=True,
                        help="name of the environment to be run (REQUIRED)")
    parser.add_argument("--model", required=True,
                        help="name of the trained model (REQUIRED)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--shift", type=int, default=0,
                        help="number of times the environment is reset at the beginning (default: 0)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="select the action with highest probability (default: False)")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="pause duration between two consequent actions of the agent (default: 0.1)")
    parser.add_argument("--gif", type=str, default=None,
                        help="store output as gif with the given filename")
    parser.add_argument("--episodes", type=int, default=1000000,
                        help="number of episodes to visualize")
    parser.add_argument("--memory", action="store_true", default=False,
                        help="add a LSTM to the model")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model")

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args.env = 'MiniGrid-DoorKey-5x5-v0'
        args.env = 'MiniGrid-KeyCorridorGBLA-v0'
        args.model = 'KeyCorridor'
        args.episodes = 100
        args.seed = 0
        args.shift = 0
        args.argmax = False
        args.memory = False
        args.text = False
        args.gif = False
        args.pause = 0.1

    if args.env == 'MiniGrid-KeyCorridorGBLA-v0':
        env_descriptor = [[0,0,0],[0,13,0],[0,0,0]]
        task_descriptor = TaskDescriptor(envD=env_descriptor,
                                         rmDesc=None,
                                         rmOrder=None,
                                         rmSize=4,
                                         observ=True,
                                         seed=None,
                                         time_steps=None)
        env = gym.make('MiniGrid-KeyCorridorGBLA-v0', taskD=task_descriptor)
        goal = GetGoalDescriptor(env)

        goal = goal.refinement[0].refinement[0].refinement[0]

        env = gym_minigrid.wrappers.FullyObsWrapper(env)
        env = gym_minigrid.wrappers.ImgObsWrapper(env)
        env = GoalRL.GoalEnvWrapper(env,goal=goal, verbose=0)
        args.env = env
    else:
        pass

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load environment

    if type(args.env) == str:
        env = utils.make_env(args.env, args.seed)
    else:
        env = args.env
    for _ in range(args.shift):
        env.reset()
    print("Environment loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                        device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text)
    print("Agent loaded\n")

    # Run the agent

    if args.gif:
       from array2gif import write_gif
       frames = []

    # Create a window to view the environment
    env.render('human')

    for episode in range(args.episodes):
        obs = env.reset()

        while True:
            env.render('human')
            if args.gif:
                frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)

            if done or env.window.closed:
                break

        if env.window.closed:
            break

    if args.gif:
        print("Saving gif... ", end="")
        write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
        print("Done.")

if __name__ == '__main__':
    main()
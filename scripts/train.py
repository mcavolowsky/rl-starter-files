import argparse
import time
import datetime
import torch
import torch_ac
import sys

import gym
import gym_minigrid

from training import TaskDescriptor
import GoalRL
from gym_minigrid.envs.goaldescriptor import *
#from stable_baselines.bench import Monitor              # stable baselines helper function for monitoring training

import utils
from model import ACModel

from copy import deepcopy


def main():
    # Parse arguments

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    ## General parameters
    parser.add_argument("--algo", required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="select the action with highest probability (default: False)")

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args.env = 'MiniGrid-DoorKey-5x5-v0'
        args.env = 'MiniGrid-KeyCorridorGBLA-v0'
        args.algo = 'ppo'
        args.seed = 1234
        args.model = 'KeyCorridor2'
        args.frames = 2e5
        args.procs = 16
        args.text = False
        args.frames_per_proc = None
        args.discount = 0.99
        args.lr = 0.001
        args.gae_lambda = 0.95
        args.entropy_coef = 0.01
        args.value_loss_coef = 0.5
        args.max_grad_norm = 0.5
        args.recurrence = 1
        args.optim_eps = 1e-8
        args.optim_alpha = 0.99
        args.clip_eps = 0.2
        args.epochs = 4
        args.batch_size = 256
        args.log_interval = 1
        args.save_interval = 10

        args.argmax = False

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

#        env = Monitor(env, 'storage/{}/{}.monitor.csv'.format(rank, goal.goalId))  # wrap the environment in the monitor object
        args.env = env
    else:
        pass


    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

    model_name = args.model or default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    # Load environments

    envs = []
    for i in range(args.procs):
        if type(args.env) == str:
            envs.append(utils.make_env(args.env, args.seed + 10000 * i))
        else:
            envs.append(deepcopy(args.env))
    txt_logger.info("Environments loaded\n")

    # Load training status


    # Load observations preprocessor

    #obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)

    # Load model

    agent = utils.Agent(envs, model_dir, logger=txt_logger,
                        argmax=args.argmax, use_memory=args.mem, use_text=args.text)

    # Load algo
    if args.algo == 'a2c':
        agent.init_training_algo(algo_type=args.algo,
                frames_per_proc=args.frames_per_proc,
                discount=args.discount,
                lr=args.lr,
                gae_lambda=args.gae_lambda,
                entropy_coef=args.entropy_coef,
                value_loss_coef=args.value_loss_coef,
                max_grad_norm=args.max_grad_norm,
                recurrence=args.recurrence,
                optim_eps=args.optim_eps,

                optim_alpha=args.optim_alpha)   # args for A2C
    elif args.algo == 'ppo':
        agent.init_training_algo(algo_type=args.algo,
                frames_per_proc=args.frames_per_proc,
                discount=args.discount,
                lr=args.lr,
                gae_lambda=args.gae_lambda,
                entropy_coef=args.entropy_coef,
                value_loss_coef=args.value_loss_coef,
                max_grad_norm=args.max_grad_norm,
                recurrence=args.recurrence,
                optim_eps=args.optim_eps,

                clip_eps=args.clip_eps,         # args for PPO2
                epochs=args.epochs,
                batch_size=args.batch_size)
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))


    agent.learn(time_steps=args.frames,
                log_interval=args.log_interval,
                save_interval=args.save_interval)

    print('training completed!')

if __name__ == '__main__':
    main()
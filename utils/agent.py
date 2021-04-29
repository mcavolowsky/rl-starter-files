"""
File: agent.py
Authors: mark cavolowsky <mark.cavolowsky@navy.mil>
Last updated: March 26, 2021

This is the module for the goal-skill agent based on the Agent class in rl-starter-files.

This is a major update to the rl-starter-files implementation to allow for an object-oriented training structure.

"""
import time                         # import standard libraries

import torch                        # import torch and torch_ac for training
import torch_ac
import tensorboardX                 # import training monitoring library

import gym                          # import gym for type-checking

import utils                        # import rl-starter-files utils (loggers, saving/loading, obs preprocessor, etc.)
from model import ACModel, MultiQModel           #

from copy import deepcopy           # import deepcopy for spawning new env instances

class Agent:
    def __init__(self, env, model_dir, model_type='multiQ', logger=None,
                 argmax=False, use_memory=False, use_text=False,
                 num_cpu=1, frames_per_proc=None,
                 discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=1, optim_eps=1e-8,
                 optim_alpha=None,
                 clip_eps=0.2, epochs=4, batch_size=256):
        """
        Initialize the Agent object.

        This primarily includes storing of the configuration parameters, but there is some other logic for correctly
        initializing the agent.

        :param env: the environment for training
        :param model_dir: the save directory (appended with the goal_id in initialization)
        :param model_type: the type of model {'PPO2', 'A2C'}
        :param logger: existing text logger
        :param argmax: if we use determinsitic or probabilistic action selection
        :param use_memory: if we are using an LSTM
        :param use_text: if we are using NLP to parse the goal
        :param num_cpu: the number of parallel instances for training
        :param frames_per_proc: max time_steps per process (versus constant)
        :param discount: the discount factor (gamma)
        :param lr: the learning rate
        :param gae_lambda: the generalized advantage estimator lambda parameter (training smoothing parameter)
        :param entropy_coef: relative weight for entropy loss
        :param value_loss_coef: relative weight for value function loss
        :param max_grad_norm: max scaling factor for the gradient
        :param recurrence: number of recurrent steps
        :param optim_eps: minimum value to prevent numerical instability
        :param optim_alpha: RMSprop decay parameter (A2C only)
        :param clip_eps: clipping parameter for the advantage and value function (PPO2 only)
        :param epochs: number of epochs in the parameter update (PPO2 only)
        :param batch_size: number of samples for the parameter update (PPO2 only)
        """
        if hasattr(env, 'goal') and env.goal:   # if the environment has a goal, set the model_dir to the goal folder
            self.model_dir = model_dir + env.goal.goalId + '/'
        else:                                   # otherwise just use the model_dir as is
            self.model_dir = model_dir

        # store all of the input parameters
        self.model_type = model_type
        self.num_cpu = num_cpu
        self.frames_per_proc = frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.optim_eps = optim_eps
        self.optim_alpha = optim_alpha
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        # use the existing logger and create two new ones
        self.txt_logger = logger
        self.csv_file, self.csv_logger = utils.get_csv_logger(self.model_dir)
        self.tb_writer = tensorboardX.SummaryWriter(self.model_dir)

        self.set_env(env)   # set the environment to with some additional checks and init of training_envs

        self.algo = None    # we don't initialize the algorithm until we call init_training_algo()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.txt_logger.info(f"Device: {device}\n")

        try:                # if we have a saved model, load it
            self.status = utils.get_status(self.model_dir)
        except OSError:     # otherwise initialize the status
            print('error loading saved model.  initializing empty model...')
            self.status = {"num_frames": 0, "update": 0}
        if self.txt_logger:self.txt_logger.info("Training status loaded\n")

        if "vocab" in self.status:
            preprocess_obss.vocab.load_vocab(self.status["vocab"])
        if self.txt_logger:self.txt_logger.info("Observations preprocessor loaded")

        # get the obs_space and the observation pre-processor
        # (for manipulating gym observations into a torch-friendly format)
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.env.observation_space)
        if model_type == 'multiQ':
            self.model = MultiQModel(obs_space, self.env.action_space,
                                     use_memory=use_memory, use_text=use_text,
                                     reward_size=2 if self.env.phi else 1)
        else:
            self.model = ACModel(obs_space, self.env.action_space, use_memory=use_memory, use_text=use_text)
        self.device = device    # store the device {'cpu', 'cuda:N'}
        self.argmax = argmax    # if we are using greedy action selection
                                # or are we using probabilistic action selection

        if self.model.recurrent:  # initialize the memories
            self.memories = torch.zeros(num_cpu, self.model.memory_size, device=self.device)

        if "model_state" in self.status:    # if we have a saved model ('model_state') in the status
                                            # load that into the initialized model
            self.model.load_state_dict(self.status["model_state"])
        self.model.to(device)             # make sure the model is located on the correct device
        self.txt_logger.info("Model loaded\n")
        self.txt_logger.info("{}\n".format(self.model))

        # some redundant code.  uncomment if there are issues and delete after enough testing
        #if 'model_state' in self.status:
        #    self.model.load_state_dict(self.status['model_state'])
        #self.model.to(self.device)
        self.model.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def init_training_algo(self, num_envs=None):
        """
        Initialize the training algorithm.

        This primarily calls the object creation functions for the A2C or PPO2 and the optimizer, but this also spawns
        a number of parallel environments, based on the self.num_cpu or num_envs input (if provided).

        Note, the spawning of parallel environments is VERY slow due to deepcopying the termination sets.  I tried some
        work arounds, but nothing worked properly, so we are stuck with it for now.

        :param num_envs: an override for the default number of environments to spawn (in self.num_cpu)
        """
        if not num_envs:
            num_envs = self.num_cpu

        self.training_envs = [deepcopy(self.env) for i in range(num_envs)]  # spawn parallel environments

        if self.model.recurrent:
            self.memories = torch.zeros(num_envs, self.model.memory_size, device=self.device)

        if self.model_type == 'multiQ':
            self.algo = torch_ac.MultiQAlgo(envs=self.training_envs,
                                        model=self.model, device=self.device,
                                        num_frames_per_proc=self.frames_per_proc,
                                        discount=self.discount, lr=self.lr,
                                        recurrence=self.recurrence,
                                        adam_eps=self.optim_eps,
                                        preprocess_obss=self.preprocess_obss)
        elif self.model_type == "A2C":
            # check to make sure that the A2C parameters are set
            assert self.optim_alpha
            self.training_envs = [deepcopy(self.env) for i in range(num_envs)]  # spawn parallel environments

            if self.model.recurrent:
                self.memories = torch.zeros(num_envs, self.model.memory_size, device=self.device)

            self.algo = torch_ac.A2CAlgo(self.training_envs,
                                         self.model, self.device,
                                         self.frames_per_proc, self.discount, self.lr, self.gae_lambda,
                                         self.entropy_coef, self.value_loss_coef, self.max_grad_norm,
                                         self.recurrence,
                                         self.optim_alpha,
                                         self.optim_eps, self.preprocess_obss)
        elif self.model_type == "PPO2":
            # check to see if the PPO2 parameters are set
            assert self.clip_eps and self.epochs and self.batch_size
            self.training_envs = [deepcopy(self.env) for i in range(num_envs)]  # spawn parallel environments

            if self.model.recurrent:
                self.memories = torch.zeros(num_envs, self.model.memory_size, device=self.device)

            self.algo = torch_ac.PPOAlgo(self.training_envs,
                                         self.model, self.device,
                                         self.frames_per_proc, self.discount, self.lr, self.gae_lambda,
                                         self.entropy_coef, self.value_loss_coef, self.max_grad_norm,
                                         self.recurrence,
                                         self.optim_eps,
                                         self.clip_eps, self.epochs, self.batch_size,
                                         self.preprocess_obss)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(algo_type))

        # load the optimizer state, if it exists
        if "optimizer_state" in self.status:
            self.algo.optimizer.load_state_dict(self.status["optimizer_state"])
        self.txt_logger.info("Optimizer loaded\n")

    def learn(self, total_timesteps, log_interval=1, save_interval=10, save_env_info=False):
        """
        The primary training loop.

        :param total_timesteps: the total number of timesteps
        :param log_interval: the period between logging/printing updates
        :param save_interval: the number of updates between model saving
        :param save_env_info: if we save the environment info (termination set) VERY SLOW
        :return: True, if training is successful
        """
        self.init_training_algo()   # initialize the training algo/environment list/optimizer

        # initialize parameters
        self.num_frames = self.status["num_frames"]
        self.update = self.status["update"]
        start_time = time.time()

        # loop until we reach the desired number of timesteps
        while self.num_frames < total_timesteps:
            # Update model parameters

            update_start_time = time.time()                 # store the time (for fps calculations)
            exps, logs1 = self.algo.collect_experiences()   # collect a number of data points for training
            logs2 = self.algo.update_parameters(exps)       # update the parameters based on the experiences
            logs = {**logs1, **logs2}                       # merge the logs for printing
            update_end_time = time.time()

            self.num_frames += logs["num_frames"]
            self.update += 1

            # all of this messy stuff is just storing and printing the log info

            if self.update % log_interval == 0:
                fps = logs["num_frames"]/(update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [self.update, self.num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                self.txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:usmM {} {} {} {} | F:usmM {} {} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | D {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if self.status["num_frames"] == 0:
                    self.csv_logger.writerow(header)
                self.csv_logger.writerow(data)
                self.csv_file.flush()

                for field, value in zip(header, data):
                    if type(value) is torch.Tensor and value.dim()>0:
                        for i,v in enumerate(value):
                            self.tb_writer.add_scalar(field+'_{}'.format(i), v, self.num_frames)
                    else:
                        self.tb_writer.add_scalar(field, value, self.num_frames)

            # Save status

            if save_interval > 0 and self.update % save_interval == 0:
                self._save_training_info()
                if save_env_info:
                    for e in self.training_envs:
                        if hasattr(e, 'save_env_info'): e.save_env_info()

        self._clear_training_envs()

        return True

    def _save_training_info(self):
        """
        Function to save the training info.
        """

        # update the status dictionary
        self.status = {"num_frames": self.num_frames, "update": self.update,
                       "model_state": self.model.state_dict(), "optimizer_state": self.algo.optimizer.state_dict()}

        if hasattr(self.preprocess_obss, "vocab"):      # if we are using NLP save, NLP info
            self.status["vocab"] = self.preprocess_obss.vocab.vocab

        utils.save_status(self.status, self.model_dir)  # save the status info to model_dir
        self.txt_logger.info("Status saved")

    def _clear_training_envs(self):
        """
        Clear the training environments to free up memory.
        """

        # the termination set gets lost, so we need to store it again
        if hasattr(self.env, 'termination_set'):
            self.env.termination_set = [s for e in self.training_envs for s in e.termination_set]

        # clear the env and the training envs
        self.algo.env = None
        if hasattr(self.env, 'termination_set'):self.training_envs = None

    def save(self, f):
        """
        Legacy function for saving the model.

        TODO: place the saving logic for the model here
        :param f:
        """
        print('self.save() - currently not implemented')

    def set_env(self, env):
        """
        Set the environment and clear the training environments

        :param env: environment for training/acting
        """
        # check to make sure the environment is the correct type
        assert isinstance(env, gym.Env)
        self.env = env
        self.training_envs = None

    def predict(self, obs, state=None, deterministic=False):
        """
        Wrapper for training code compatibility.  Calls get_action() to predict the action to take based on the
        current observation.

        :param obs: observation for predicting the action
        :param state: state of the LSTM (unused)
        :param deterministic: whether to use deterministic or probabilistic actions (unused)
        :return: action and LSTM state
        """
        # assert (state==None) and (deterministic==False) # still need to reimplement
        return self.get_action(obs), None   # return action, states - states is unused at the moment

    def get_actions(self, obss):
        """
        Get a list of actions for a list of observations.



        :param obss: list of observations for predicting actions
        :return: list of actions for the associated observations
        """
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():                   # don't calculate the gradients, since we are doing a forward pass
            if self.model.recurrent:          # if we are using a recurrent model
                dist, _, self.memories = self.model(preprocessed_obss, self.memories)
            else:                               # otherwise
                dist, _ = self.model(preprocessed_obss)
                                                # preprocess the observations to put them in a torch-friendly format

        # the acmodel returns a probability distribution
        if self.argmax:                         # if we are detemrinistic, take the action with the highest probability
            actions = dist.probs.max(1, keepdim=True)[1]
        else:                                   # otherwise sample the distribution to select the action
            actions = dist.sample()

        return actions.cpu().numpy()            # reaturn a numpy array, not a tensor

    def get_action(self, obs):
        """
        Wrapper for get_actions() to produce just a single action (rather than a list of actions) for acting.

        :param obs: single observation
        :return: single action
        """
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        """
        rl-starter-files code.  Not sure what this does.

        :param rewards:
        :param dones:
        """
        if self.model.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        """
        rl-starter-files code.  Not sure what this does (other than wrap analyze_feedbacks().

        :param reward:
        :param done:
        :return:
        """
        return self.analyze_feedbacks([reward], [done])

import time

import torch
import torch_ac
import tensorboardX

import gym

import utils
from model import ACModel

from copy import deepcopy

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, env, model_dir, logger=None,
                 argmax=False, use_memory=False, use_text=False):
        self.txt_logger = logger
        if env.goal:
            self.model_dir = model_dir + env.goal.goalId + '/'
        else:
            self.model_dir = model_dir

        self.csv_file, self.csv_logger = utils.get_csv_logger(self.model_dir)
        self.tb_writer = tensorboardX.SummaryWriter(self.model_dir)

        self.set_env(env)

        self.algo = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.txt_logger.info(f"Device: {device}\n")

        try:
            self.status = utils.get_status(self.model_dir)
        except OSError:
            self.status = {"num_frames": 0, "update": 0}
        if self.txt_logger:self.txt_logger.info("Training status loaded\n")

        if "vocab" in self.status:
            preprocess_obss.vocab.load_vocab(self.status["vocab"])
        if self.txt_logger:self.txt_logger.info("Observations preprocessor loaded")

        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(self.env.observation_space)
        self.acmodel = ACModel(obs_space, self.env.action_space, use_memory=use_memory, use_text=use_text)
        self.device = device
        self.argmax = argmax

        if "model_state" in self.status:
            self.acmodel.load_state_dict(self.status["model_state"])
        self.acmodel.to(device)
        self.txt_logger.info("Model loaded\n")
        self.txt_logger.info("{}\n".format(self.acmodel))

        if 'model_state' in self.status:
            self.acmodel.load_state_dict(self.status['model_state'])
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def init_training_algo(self, algo_type, num_cpu,
                    frames_per_proc, discount, lr, gae_lambda,
                    entropy_coef, value_loss_coef, max_grad_norm, recurrence, optim_eps,
                    optim_alpha=None,
                    clip_eps=None, epochs=None, batch_size=None):
        if algo_type == "a2c":
            assert optim_alpha
            self.training_envs = [deepcopy(self.env) for i in range(num_cpu)]

            if self.acmodel.recurrent:
                self.memories = torch.zeros(num_cpu, self.acmodel.memory_size, device=self.device)

            self.algo = torch_ac.A2CAlgo(self.training_envs,
                                    self.acmodel, self.device,
                                    frames_per_proc, discount, lr, gae_lambda,
                                    entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                                    optim_alpha, optim_eps, self.preprocess_obss)
        elif algo_type == "ppo":
            assert clip_eps and epochs and batch_size
            self.training_envs = [deepcopy(self.env) for i in range(num_cpu)]

            if self.acmodel.recurrent:
                self.memories = torch.zeros(num_cpu, self.acmodel.memory_size, device=self.device)

            self.algo = torch_ac.PPOAlgo(self.training_envs,
                                    self.acmodel, self.device,
                                    frames_per_proc, discount, lr, gae_lambda,
                                    entropy_coef, value_loss_coef, max_grad_norm, recurrence,
                                    optim_eps, clip_eps, epochs, batch_size, self.preprocess_obss)
        else:
            raise ValueError("Incorrect algorithm name: {}".format(algo_type))

        if "optimizer_state" in self.status:
            self.algo.optimizer.load_state_dict(self.status["optimizer_state"])
        self.txt_logger.info("Optimizer loaded\n")

    def learn(self, total_timesteps, log_interval=1, save_interval=10):
        self.num_frames = self.status["num_frames"]
        update = self.status["update"]
        start_time = time.time()

        while self.num_frames < total_timesteps:
            # Update model parameters

            update_start_time = time.time()
            exps, logs1 = self.algo.collect_experiences()
            logs2 = self.algo.update_parameters(exps)
            logs = {**logs1, **logs2}
            update_end_time = time.time()

            self.num_frames += logs["num_frames"]
            update += 1

            # Print logs

            if update % log_interval == 0:
                fps = logs["num_frames"]/(update_end_time - update_start_time)
                duration = int(time.time() - start_time)
                return_per_episode = utils.synthesize(logs["return_per_episode"])
                rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
                num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

                header = ["update", "frames", "FPS", "duration"]
                data = [update, self.num_frames, fps, duration]
                header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
                data += rreturn_per_episode.values()
                header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
                data += num_frames_per_episode.values()
                header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
                data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

                self.txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:usmM {:.2f} {:.2f} {:.2f} {:.2f} | F:usmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | D {:.3f}"
                    .format(*data))

                header += ["return_" + key for key in return_per_episode.keys()]
                data += return_per_episode.values()

                if self.status["num_frames"] == 0:
                    self.csv_logger.writerow(header)
                self.csv_logger.writerow(data)
                self.csv_file.flush()

                for field, value in zip(header, data):
                    self.tb_writer.add_scalar(field, value, self.num_frames)

            # Save status

            if save_interval > 0 and update % save_interval == 0:
                self._save_training_info(update)

        # the termination set gets lost, so we need to store it again
        self.env.termination_set = [s for e in self.training_envs for s in e.termination_set]
        return True

    def _save_training_info(self, update):
        self.status = {"num_frames": self.num_frames, "update": update,
                       "model_state": self.acmodel.state_dict(), "optimizer_state": self.algo.optimizer.state_dict()}
        if hasattr(self.preprocess_obss, "vocab"):
            self.status["vocab"] = self.preprocess_obss.vocab.vocab
        utils.save_status(self.status, self.model_dir)
        self.txt_logger.info("Status saved")

    def save(self, f):
        print('self.save() - currently not implemented')

    def set_env(self, env):
        assert isinstance(env, gym.Env)
        self.env = env
        self.training_envs = None

    def predict(self, obss, state=None, deterministic=False):
        return self.get_actions(obss)

    def setup_model(self):
        print('self.setup_model() - currently not implemented')

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

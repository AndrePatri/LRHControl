from lrhc_control.agents.ppo_agent import Agent

import torch 
import torch.optim as optim
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import random

from typing import Dict

import os

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
        
class CleanPPO():

    def __init__(self,
            env):

        self._env = env 

        self._agent = Agent(obs_dim=self._env.obs_dim(),
                        actions_dim=self._env.actions_dim())
        
        self._optimizer = None

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._model_path = None

        self._hyperparameters = {}
        
        self._init_params()

        self._setup_done = False

        self._verbose = False
              
    def setup(self,
            run_name: str,
            custom_args: Dict = {},
            verbose: bool = False):
        
        self._run_name = run_name
        self._hyperparameters = custom_args
        
        self._verbose = verbose

        self._drop_dir = "./" + f"{self.__class__.__name__}/" + self._run_name
        self._model_path = self._drop_dir + "model"
        model_directory = os.path.dirname(self._model_path)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # seeding
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.backends.cudnn.deterministic = self._torch_deterministic

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")

        self._agent.to(self._torch_device) # move agent to target device

        self._optimizer = optim.Adam(self._agent.parameters(), 
                                lr=self._learning_rate, 
                                eps=1e-5 # small constant added to the optimization
                                )
        self._init_buffers()

        self._setup_done = True
       
    def step(self):
        
        if not self._setup_done:
        
            self._should_have_called_setup()

        # annealing the rate if enabled
        if self._anneal_lr:

            frac = 1.0 - (self._it_counter - 1.0) / self._iterations_n
            lrnow = frac * self._learning_rate
            self._optimizer.param_groups[0]["lr"] = lrnow

        # collect data from current policy during an episode
        for step in range(self._episode_n_steps):

            # sample actions from agent
            with torch.no_grad(): # no need for gradients computation
                action, logprob, _, value = self._agent.get_action_and_value(self._env.get_last_obs())
                self._values[step] = value.reshape(-1, 1)
                self._actions[step] = action.reshape(-1, 1)
                self._logprobs[step] = logprob.reshape(-1, 1)
            
            # step vectorized env
            self._env.step(action) 

            # retrieve termination states and rewards

            self._obs[step] = self._env.get_last_obs()
            self._rewards[step] = self._env.get_last_rewards()
            self._dones[step] = torch.logical_or(self._env.get_last_terminations(),
                                            self._env.get_last_truncations()) # either terminated or truncated

        # bootstrap value if not done
        with torch.no_grad():

            self._advantages.zero_() # reset advantages

            lastgaelam = 0
            for t in reversed(range(self._episode_n_steps)):

                if t == self._episode_n_steps - 1: # last step

                    nextnonterminal = 1.0 - self._dones[t]

                    nextvalues = self._values[t]

                else:

                    nextnonterminal = 1.0 - self._dones[t + 1]

                    nextvalues = self._values[t + 1]

                # temporal difference error computation
                actual_reward_discounted = self._rewards[t] + self._discount_factor * nextvalues * nextnonterminal
                td_error = actual_reward_discounted - self._values[t] # meas. - est. reward

                # compute advantages using the Generalized Advantage Estimation (GAE) 
                self._advantages[t] = lastgaelam = td_error + self._discount_factor * self._gae_lambda * nextnonterminal * lastgaelam

            # estimated cumulative rewards from each time step to the end of the episode
            self._returns[:, :] = self._advantages + self._values

        # flatten batches
        batched_obs = self._obs.reshape((-1, self._env.obs_dim()))
        batched_logprobs = self._logprobs.reshape(-1)
        batched_actions = self._actions.reshape((-1, self._env.actions_dim()))
        batched_advantages = self._advantages.reshape(-1)
        batched_returns = self._returns.reshape(-1)
        batched_values = self._values.reshape(-1)

        # optimize policy and value network
        clipfracs = []

        for epoch in range(self._update_epochs):

            shuffled_batch_indxs = torch.randperm(self._batch_size) # randomizing 
            # indexes for removing correlations

            for start in range(0, self._batch_size, self._minibatch_size):
                
                end = start + self._minibatch_size
                minibatch_inds = shuffled_batch_indxs[start:end]

                _, newlogprob, entropy, newvalue = self._agent.get_action_and_value(
                                                                    batched_obs[minibatch_inds], 
                                                                    batched_actions[minibatch_inds])
                logratio = newlogprob - batched_logprobs[minibatch_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

                minibatch_advantages = batched_advantages[minibatch_inds]
                if self._norm_adv:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self._clip_vloss:
                    v_loss_unclipped = (newvalue - batched_returns[minibatch_inds]) ** 2
                    v_clipped = batched_values[minibatch_inds] + torch.clamp(
                        newvalue - batched_values[minibatch_inds],
                        -self._clip_coef,
                        self._clip_coef,
                    )
                    v_loss_clipped = (v_clipped - batched_returns[minibatch_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - batched_returns[minibatch_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self._entropy_coeff * entropy_loss + v_loss * self._val_f_coeff

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._agent.parameters(), self._max_grad_norm)
                self._optimizer.step()
            
            if self._target_kl is not None and approx_kl > self._target_kl:

                break

        self._post_step()

    def _post_step(self):

        self._it_counter +=1 
        
        if self._verbose:

            info = f"Current step n.{self._it_counter + 1}/{self._episode_n_steps}"

            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if self._it_counter == self._iterations_n:

            self._done()

            exit()
 
    def _done(self):

        if self._save_model:

            torch.save(self._agent.state_dict(), self._model_path)

    # def _evaluate(self,
    #         eval_episodes: int,
    #         run_name: str,
    #         device: torch.device = torch.device("cpu"),
    #         capture_video: bool = True,
    #         gamma: float = 0.99,
    #     ):

    #     self._agent.load_state_dict(torch.load(self._model_path, map_location=self._torch_device))

    #     self._agent.eval()

    #     self._env.reset()

    #     episodic_returns = []
    #     while len(episodic_returns) < eval_episodes:
    #         actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
    #         next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
    #         if "final_info" in infos:
    #             for info in infos["final_info"]:
    #                 if "episode" not in info:
    #                     continue
    #                 print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
    #                 episodic_returns += [info["episode"]["r"]]
    #         obs = next_obs

    #     return episodic_returns

    def _should_have_called_setup(self):

        exception = f"setup() was not called!"

        Journal.log(self.__class__.__name__,
            "_should_have_called_setup",
            exception,
            LogType.EXCEP,
            throw_when_excep = True)
        
    def _init_params(self):

        self._dtype = self._env.dtype()

        self._seed = 1
        self._run_name = "DummyRun"
        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu

        self._torch_deterministic = True
        self._save_model = True
        self._env_name = self._env.name()

        self._total_timesteps = 10000000
        self._learning_rate = 3e-4

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._episode_n_steps = 4096
        self._anneal_lr = True
        self._discount_factor = 0.99
        self._gae_lambda = 0.95
        self._num_minibatches = 32
        self._update_epochs = 10
        self._norm_adv = True
        self._clip_coef = 0.2
        self._clip_vloss = True
        self._entropy_coeff = 0.0
        self._val_f_coeff = 0.5
        self._max_grad_norm = 0.5
        self._target_kl = None

        self._batch_size =int(self._num_envs * self._episode_n_steps)
        self._minibatch_size = int(self._batch_size // self._num_minibatches)
        self._iterations_n = self._total_timesteps // self._batch_size
        
        self._it_counter = 0
    
    def _init_buffers(self):

        self._obs = torch.full(size=(self._episode_n_steps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._actions = torch.full(size=(self._episode_n_steps, self._num_envs, self._actions_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._logprobs = torch.full(size=(self._episode_n_steps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._rewards = torch.full(size=(self._episode_n_steps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._dones = torch.full(size=(self._episode_n_steps, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._values = torch.full(size=(self._episode_n_steps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._advantages = torch.full(size=(self._episode_n_steps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._returns = torch.full(size=(self._episode_n_steps, self._num_envs, 1),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        
    def _init_writer(self):

        self._writer = SummaryWriter(f"runs/{self._run_name}")
        self._writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self._hyperparameters).items()])),
        )
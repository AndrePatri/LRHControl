from lrhc_control.agents.ppo_agent import Agent

import torch 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import random

from typing import Dict

class CleanPPO():

    def __init__(self,
            env):

        self._env = env 

        self._agent = Agent(obs_dim=self._env.obs_dim(),
                        actions_dim=self._env.actions_dim())

        self._optimizer = None

        self._writer = None
        
        self._run_name = ""

        self._custom_args = {}
        
        self._init_params()

    def setup(self,
            run_name: str,
            custom_args: Dict = {}):
        
        self._run_name = run_name
        self._custom_args = custom_args
        
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
       
    def step(self):

        # annealing the rate if enabled
        if self._anneal_lr:

            frac = 1.0 - (self._it_counter - 1.0) / self._iterations_n
            lrnow = frac * self._learning_rate
            self._optimizer.param_groups[0]["lr"] = lrnow

        # collect data from current policy
        for step in range(self._num_steps):

            # sample actions from agent
            with torch.no_grad(): # no need for gradients computation
                action, logprob, _, value = self._agent.get_action_and_value(self._env.get_last_obs())
                self._values[step] = value.flatten()
                self._actions[step] = action
                self._logprobs[step] = logprob
            
            # step vectorized env
            self._env.step(action) 

            # retrieve termination states and rewards
            self._dones[step] = torch.logical_or(self._env.get_last_terminations(),
                                            self._env.get_last_truncations())
            self._obs[step] = self._env.get_last_obs()
            self._rewards[step] = self._env.get_last_rewards()

        # bootstrap value if not done
        with torch.no_grad():

            next_value = self._agent.get_value(self._env.get_last_obs()).reshape(1, -1)
            self._advantages.zero_() # reset advantages
            lastgaelam = 0

            for t in reversed(range(self._num_steps)):

                if t == self._num_steps - 1: # last step

                    nextnonterminal = 1.0 - self._dones[t]

                    nextvalues = next_value

                else:

                    nextnonterminal = 1.0 - self._dones[t + 1]

                    nextvalues = self._values[t + 1]

                delta = self._rewards[t] + self._discount_factor * nextvalues * nextnonterminal - self._values[t]

                self._advantages[t] = lastgaelam = delta + self._discount_factor * self._gae_lambda * nextnonterminal * lastgaelam

            self._returns[:, :] = self._advantages + self._values

        # flatten batch
        batched_obs = self._obs.reshape((-1, self._env.obs_dim()))
        batched_logprobs = self._logprobs.reshape(-1)
        batched_actions = self._actions.reshape((-1, self._env.actions_dim()))
        batched_advantages = self._advantages.reshape(-1)
        batched_returns = self._returns.reshape(-1)
        batched_values = self._values.reshape(-1)

        # optimize policy and value network
        batch_indxs = torch.arange(self._batch_size)
        clipfracs = []

        for epoch in range(self._update_epochs):
            torch.random.shuff

        self._post_step()

    def _post_step(self):

        self._it_counter +=1 
        
        if self._it_counter == self._iterations_n:

            self._done()

            exit()
 
    def _done(self):

        a = 2

    def _init_params(self):

        self._dtype = self._env.dtype()

        self._seed = 1
        self._run_name = "DummyRun"
        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu

        self._torch_deterministic = True
        self._save_model = True
        self._env_name = self._env.name()

        self._total_timesteps = 1000000
        self._learning_rate = 3e-4

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._num_steps = 2048
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

        self._batch_size =int(self._num_envs * self._num_steps)
        self._minibatch_size = int(self._batch_size // self._num_minibatches)
        self._iterations_n = self._total_timesteps // self._batch_size

        self._it_counter = 0
    
    def _init_buffers(self):

        self._obs = torch.full(size=(self._num_steps, self._num_envs, self._obs_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._actions = torch.full(size=(self._num_steps, self._num_envs, self._actions_dim),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._logprobs = torch.full(size=(self._num_steps, self._num_envs),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._rewards = torch.full(size=(self._num_steps, self._num_envs),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._dones = torch.full(size=(self._num_steps, self._num_envs),
                        fill_value=False,
                        dtype=self._dtype,
                        device=torch.bool_)
        self._values = torch.full(size=(self._num_steps, self._num_envs),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._advantages = torch.full(size=(self._num_steps, self._num_envs),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._returns = torch.full(size=(self._num_steps, self._num_envs),
                        fill_value=0,
                        dtype=self._dtype,
                        device=self._torch_device)
        
    def _init_writer(self):

        self._writer = SummaryWriter(f"runs/{self._run_name}")
        self._writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self._custom_args).items()])),
        )
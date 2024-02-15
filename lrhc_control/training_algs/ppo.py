from lrhc_control.agents.ppo_agent import Agent

class CleanPPO():

    def __init__(self,
            env):

        self._env = env

        self._agent = Agent(obs_dim=self._env.obs_dim(),
                        actions_dim=self._env.actions_dim())
    
    def _check_env(self):

        a = 1
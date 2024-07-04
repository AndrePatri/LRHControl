from lrhc_control.agents.sactor_critic.sac import SACAgent

from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo
import torch 
import torch.optim as optim
import torch.nn as nn

import random

from typing import Dict

import os
import shutil

import time

import wandb

from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import VLevel

from abc import ABC, abstractmethod

class SActorCriticAlgoBase():

    # base class for actor-critic RL algorithms
     
    def __init__(self,
            env, 
            debug = False,
            remote_db = False,
            anomaly_detect = False,
            seed: int = 1):

        self._env = env 
        self._seed = seed

        self._eval = False
        self._agent = None 
        
        self._debug = debug
        self._remote_db = remote_db

        self._anomaly_detect = anomaly_detect

        self._optimizer = None

        self._writer = None
        
        self._run_name = None
        self._drop_dir = None
        self._dbinfo_drop_fname = None
        self._model_path = None
        
        self._policy_update_db_data_dict =  {}
        self._custom_env_data_db_dict = {}
        self._hyperparameters = {}
        
        self._episodic_reward_getter = self._env.ep_reward_getter()
        
        self._init_params()
        
        self._init_dbdata()

        self._setup_done = False

        self._verbose = False

        self._is_done = False
        
        self._shared_algo_data = None

        self._this_child_path = None
        self._this_basepath = os.path.abspath(__file__)
    
    def __del__(self):

        self.done()

    def learn(self):

        if not self._setup_done:
            self._should_have_called_setup()

        self._start_time = time.perf_counter()

        if not self._collect_transition():
            return False
        
        self._collection_t = time.perf_counter()
        
        self._update_policy()

        self._policy_update_t = time.perf_counter()

        self._post_step()

        return True

    def eval(self):

        if not self._setup_done:
            self._should_have_called_setup()

        self._start_time = time.perf_counter()

        if not self._collect_transition():
            return False

        self._collection_t = time.perf_counter()
        
        self._post_step()

        return True
    
    @abstractmethod
    def _collect_transition(self)->bool:
        pass
    
    @abstractmethod
    def _update_policy(self):
        pass

    def setup(self,
            run_name: str,
            custom_args: Dict = {},
            verbose: bool = False,
            drop_dir_name: str = None,
            eval: bool = False,
            model_path: str = None,
            n_evals: int = None,
            n_timesteps_per_eval: int = None,
            comment: str = "",
            dump_checkpoints: bool = False,
            norm_obs: bool = True):

        self._verbose = verbose

        self._dump_checkpoints = dump_checkpoints
        
        self._eval = eval

        self._run_name = run_name
        from datetime import datetime
        self._time_id = datetime.now().strftime('d%Y_%m_%d_h%H_m%M_s%S')
        self._unique_id = self._time_id + "-" + self._run_name
        self._init_algo_shared_data(static_params=self._hyperparameters) # can only handle dicts with
        # numeric values
        self._hyperparameters.update(custom_args)

        self._torch_device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")

        self._agent = SACAgent(obs_dim=self._env.obs_dim(),
                    actions_dim=self._env.actions_dim(),
                    actions_ub=self._env.get_actions_ub().flatten().tolist(),
                    actions_lb=self._env.get_actions_lb().flatten().tolist(),
                    norm_obs=norm_obs,
                    device=self._torch_device,
                    dtype=self._dtype,
                    is_eval=self._eval
                    )

        # load model if necessary 
        if self._eval: # load pretrained model
            if model_path is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, a model_path should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            elif n_timesteps_per_eval is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, n_timesteps_per_eval should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            elif n_evals is None:
                Journal.log(self.__class__.__name__,
                    "setup",
                    f"When eval is True, n_evals should be provided!!",
                    LogType.EXCEP,
                    throw_when_excep = True)
            else:
                self._model_path = model_path
                self._total_timesteps = n_evals
            self._load_model(self._model_path)

        # create dump directory + copy important files for debug
        self._init_drop_dir(drop_dir_name)
            
        # seeding + deterministic behavior for reproducibility
        self._set_all_deterministic()
        torch.autograd.set_detect_anomaly(self._anomaly_detect)

        if (self._debug):
            if self._remote_db:
                job_type = "evaluation" if self._eval else "training"
                wandb.init(
                    project="LRHControl",
                    group=self._run_name,
                    name=self._unique_id,
                    id=self._unique_id,
                    job_type=job_type,
                    # tags=None,
                    notes=comment,
                    resume="never", # do not allow runs with the same unique id
                    mode="online", # "online", "offline" or "disabled"
                    entity=None,
                    sync_tensorboard=True,
                    config=self._hyperparameters,
                    monitor_gym=True,
                    save_code=True,
                    dir=self._drop_dir
                )
                wandb.watch(self._agent, log="all")

        if not self._eval:
            self._qf_optimizer = optim.Adam(list(self._agent.qf1.parameters()) + list(self._agent.qf2.parameters()), 
                                    lr=self._lr_q)
            self._actor_optimizer = optim.Adam(list(self._agent.actor.parameters()), 
                                    lr=self._lr_policy)

        self._init_replay_buffers()
        
        if self._autotune:
            self._target_entropy = -self._env.actions_dim()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._torch_device)
            self._alpha = self._log_alpha.exp().item()
            self._a_optimizer = optim.Adam([self._log_alpha], lr=self._lr_q)
    
        # self._env.reset()
        
        self._setup_done = True

        self._is_done = False

        self._start_time_tot = time.perf_counter()

        self._start_time = time.perf_counter()
    
    def is_done(self):

        return self._is_done 
    
    def model_path(self):

        return self._model_path

    def _save_model(self,
            is_checkpoint: bool = False):

        path = self._model_path
        if is_checkpoint: # use iteration as id
            path = path + "_checkpoint" + str(self._log_it_counter)
        info = f"Saving model to {path}"
        Journal.log(self.__class__.__name__,
            "done",
            info,
            LogType.INFO,
            throw_when_excep = True)
        torch.save(self._agent.state_dict(), path) # saves whole agent state
        # torch.save(self._agent.parameters(), path) # only save agent parameters
        info = f"Done."
        Journal.log(self.__class__.__name__,
            "done",
            info,
            LogType.INFO,
            throw_when_excep = True)
                    
    def done(self):
        
        if not self._is_done:

            if not self._eval:
                self._save_model()
            
            self._dump_dbinfo_to_file()
            
            if self._shared_algo_data is not None:
                self._shared_algo_data.close() # close shared memory

            self._env.close()

            self._is_done = True

    def _dump_dbinfo_to_file(self):

        import h5py

        info = f"Dumping debug info at {self._dbinfo_drop_fname}"
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        with h5py.File(self._dbinfo_drop_fname+".hdf5", 'w') as hf:
            # hf.create_dataset('numpy_data', data=numpy_data)
            # Write dictionaries to HDF5 as attributes
            for key, value in self._hyperparameters.items():
                if value is None:
                    value = "None"
                hf.attrs[key] = value
            
            # rewards
            hf.create_dataset('sub_reward_names', data=self._reward_names, 
                dtype='S20') 
            hf.create_dataset('episodic_rewards', data=self._episodic_rewards.numpy())
            hf.create_dataset('episodic_sub_rewards', data=self._episodic_sub_rewards.numpy())
            hf.create_dataset('episodic_sub_rewards_env_avrg', data=self._episodic_sub_rewards_env_avrg.numpy())
            hf.create_dataset('episodic_rewards_env_avrg', data=self._episodic_rewards_env_avrg.numpy())

            # profiling data
            hf.create_dataset('env_step_fps', data=self._env_step_fps.numpy())
            hf.create_dataset('env_step_rt_factor', data=self._env_step_rt_factor.numpy())
            hf.create_dataset('policy_update_dt', data=self._policy_update_dt.numpy())
            hf.create_dataset('policy_update_fps', data=self._policy_update_fps.numpy())
            hf.create_dataset('n_of_played_episodes', data=self._n_of_played_episodes.numpy())
            hf.create_dataset('n_timesteps_done', data=self._n_timesteps_done.numpy())
            hf.create_dataset('n_policy_updates', data=self._n_policy_updates.numpy())
            hf.create_dataset('elapsed_min', data=self._elapsed_min.numpy())

            # dump all custom env data
            db_data_names = list(self._env.custom_db_data.keys())
            for db_dname in db_data_names:
                data=self._custom_env_data[db_dname]
                subnames = list(data.keys())
                for subname in subnames:
                    var_name = db_dname + "_" + subname
                    hf.create_dataset(var_name, data=data[subname])
            db_info_names = list(self._env.custom_db_info.keys())
            for db_info in db_info_names:
                hf.create_dataset(db_info, data=self._env.custom_db_info[db_info])
        info = f"done."
        Journal.log(self.__class__.__name__,
            "_dump_dbinfo_to_file",
            info,
            LogType.INFO,
            throw_when_excep = True)

    def _load_model(self,
            model_path: str):
        
        info = f"Loading model at {model_path}"

        Journal.log(self.__class__.__name__,
            "_load_model",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._agent.load_state_dict(torch.load(model_path, 
                            map_location=self._torch_device))
        self._agent.eval()

    def _set_all_deterministic(self):

        random.seed(self._seed) # python seed
        torch.manual_seed(self._seed)
        torch.backends.cudnn.deterministic = self._torch_deterministic
        # torch.use_deterministic_algorithms(mode=True) # will throw excep. when trying to use non-det. algos
        import numpy as np
        np.random.seed(0)

    def _init_drop_dir(self,
                drop_dir_name: str = None):

        # main drop directory
        if drop_dir_name is None:
            # drop to current directory
            self._drop_dir = "./" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id
        else:
            self._drop_dir = drop_dir_name + "/" + f"{self.__class__.__name__}/" + self._run_name + "/" + self._unique_id
        os.makedirs(self._drop_dir)

        # model
        if not self._eval:
            self._model_path = self._drop_dir + "/" + self._unique_id + "_model"
        else: # we copy the model under evaluation to the drop dir
            shutil.copy(self._model_path, self._drop_dir)

        # debug info
        self._dbinfo_drop_fname = self._drop_dir + "/" + self._unique_id + "db_info" # extension added later

        # other auxiliary db files
        aux_drop_dir = self._drop_dir + "/other"
        os.makedirs(aux_drop_dir)
        filepaths = self._env.get_file_paths() # envs implementation
        filepaths.append(self._this_basepath) # algorithm implementation
        filepaths.append(self._this_child_path)
        filepaths.append(self._agent.get_impl_path()) # agent implementation
        for file in filepaths:
            shutil.copy(file, self._drop_dir)
        aux_dirs = self._env.get_aux_dir()
        for aux_dir in aux_dirs:
            shutil.copytree(aux_dir, aux_drop_dir, dirs_exist_ok=True)

    def _post_step(self):
        
        self._vec_transition_counter+=1
        
        self._collection_dt[self._log_it_counter] += \
            (self._collection_t-self._start_time) 
        self._policy_update_dt[self._log_it_counter] += \
            (self._policy_update_t - self._collection_t)
        
        if self._vec_transition_counter % self._db_vecstep_frequency== 0:
            # only log data every n timesteps 
        
            self._env_step_fps[self._log_it_counter] = (self._db_vecstep_frequency*self._num_envs)/ self._collection_dt[self._log_it_counter]
            self._env_step_rt_factor[self._log_it_counter] = self._env_step_fps[self._log_it_counter] * self._hyperparameters["control_clust_dt"]

            self._n_of_played_episodes[self._log_it_counter] = self._episodic_reward_getter.get_n_played_episodes()
            self._n_timesteps_done[self._log_it_counter] = self._vec_transition_counter

            self._n_policy_updates[self._log_it_counter] = self._vec_transition_counter*self._policy_freq

            self._policy_update_fps[self._log_it_counter] = self._n_policy_updates[self._log_it_counter]/self._policy_update_dt[self._log_it_counter]

            self._elapsed_min[self._log_it_counter] = (time.perf_counter() - self._start_time_tot) / 60
        
            # we get some episodic reward metrics
            self._episodic_rewards[self._log_it_counter, :, :] = self._episodic_reward_getter.get_rollout_avrg_total_reward() # total ep. rewards across envs
            self._episodic_rewards_env_avrg[self._log_it_counter, :, :] = self._episodic_reward_getter.get_rollout_avrg_total_reward_env_avrg() # tot, avrg over envs
            self._episodic_sub_rewards[self._log_it_counter, :, :] = self._episodic_reward_getter.get_rollout_avrg_reward() # sub-episodic rewards across envs
            self._episodic_sub_rewards_env_avrg[self._log_it_counter, :, :] = self._episodic_reward_getter.get_rollout_reward_env_avrg() # avrg over envs
            self._episodic_reward_getter.reset() # necessary, we don't want to accumulate 
            # debug rewards from previous db iterations

            # fill env custom db metrics
            db_data_names = list(self._env.custom_db_data.keys())
            for dbdatan in db_data_names:
                self._custom_env_data[dbdatan]["rollout_stat"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat()
                self._custom_env_data[dbdatan]["rollout_stat_env_avrg"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat_env_avrg()
                self._custom_env_data[dbdatan]["rollout_stat_comp"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat_comp()
                self._custom_env_data[dbdatan]["rollout_stat_comp_env_avrg"][self._log_it_counter, :, :] = self._env.custom_db_data[dbdatan].get_rollout_stat_comp_env_avrg()
            self._env.reset_custom_db_data() # reset custom db stats for this iteration

            self._log_info()

            self._log_it_counter+=1 

        if self._dump_checkpoints and \
            (self._vec_transition_counter % self._m_checkpoint_freq == 0):
            self._save_model(is_checkpoint=True)

        if self._vec_transition_counter == self._total_timesteps_vec:
            self.done()            
            
    def _should_have_called_setup(self):

        exception = f"setup() was not called!"

        Journal.log(self.__class__.__name__,
            "_should_have_called_setup",
            exception,
            LogType.EXCEP,
            throw_when_excep = True)
    
    def _log_info(self):
        
        if self._debug:
            # add custom env db data (dumped LOCALLY at the end of training)
            db_data_names = list(self._env.custom_db_data.keys())
            for dbdatan in db_data_names: 
                data = self._custom_env_data[dbdatan]
                data_names = self._env.custom_db_data[dbdatan].data_names()
                self._custom_env_data_db_dict.update({f"{dbdatan}" + "_rollout_stat_comp": 
                        wandb.Histogram(data["rollout_stat_comp"][self._log_it_counter-1, :, :].numpy())})
                self._custom_env_data_db_dict.update({f"{dbdatan}" + "_rollout_stat_comp_env_avrg": 
                        data["rollout_stat_comp_env_avrg"][self._log_it_counter-1, :, :].item()})
                self._custom_env_data_db_dict.update({f"sub_env_dbdata/{dbdatan}-{data_names[i]}" + "_rollout_stat_env_avrg": 
                       data["rollout_stat_env_avrg"][self._log_it_counter-1, :, i:i+1] for i in range(len(data_names))})
                self._custom_env_data_db_dict.update({f"sub_env_dbdata/{dbdatan}-{data_names[i]}" + "_rollout_stat": 
                        wandb.Histogram(data["rollout_stat"].numpy()[self._log_it_counter-1, :, i:i+1]) for i in range(len(data_names))})

            if self._remote_db: 
                # write general algo debug info to shared memory    
                info_names=["current_ppo_iteration", 
                    "n_of_performed_policy_updates",
                    "n_of_played_episodes", 
                    "n_of_timesteps_done",
                    "env_step_fps",
                    "env_step_rt_factor",
                    "policy_improv_fps",
                    "elapsed_min"
                    ]
                info_data = [self._log_it_counter, 
                    self._n_policy_updates[self._log_it_counter].item(),
                    self._n_of_played_episodes[self._log_it_counter].item(), 
                    self._n_timesteps_done[self._log_it_counter].item(),
                    self._policy_update_dt[self._log_it_counter].item(),
                    self._env_step_fps[self._log_it_counter].item(),
                    self._env_step_rt_factor[self._log_it_counter].item(),
                    self._policy_update_fps[self._log_it_counter].item(),
                    self._elapsed_min[self._log_it_counter].item()
                    ]
                # self._shared_algo_data.write(dyn_info_name=info_names,
                #                         val=info_data)

                # write debug info to remote wandb server
                wandb_d = {'tot_episodic_reward': wandb.Histogram(self._episodic_rewards[self._log_it_counter-1, :, :].numpy()),
                    'tot_episodic_reward_env_avrg': self._episodic_rewards_env_avrg[self._log_it_counter-1, :, :].item(),
                    'ppo_iteration' : self._log_it_counter}
                wandb_d.update(dict(zip(info_names, info_data)))
                wandb_d.update({f"sub_reward/{self._reward_names[i]}_env_avrg":
                        self._episodic_sub_rewards_env_avrg[self._log_it_counter-1, :, i:i+1] for i in range(len(self._reward_names))})
                wandb_d.update({f"sub_reward/{self._reward_names[i]}":
                        wandb.Histogram(self._episodic_sub_rewards.numpy()[self._log_it_counter-1, :, i:i+1]) for i in range(len(self._reward_names))})
                
                wandb_d.update(self._policy_update_db_data_dict)
                wandb_d.update(self._custom_env_data_db_dict)

                wandb.log(wandb_d)

        if self._verbose:
            
            elapsed_h = self._elapsed_min[self._log_it_counter].item()/60.0
            est_remaining_time_h =  elapsed_h * 1/self._vec_transition_counter * (self._total_timesteps_vec-self._vec_transition_counter)

            info =f"\nTotal n. timesteps simulated: {self._vec_transition_counter*self._num_envs}/{self._total_timesteps}\n" + \
                f"Elapsed time: {self._elapsed_min[self._log_it_counter].item()/60.0} h\n" + \
                f"Estimated remaining training time: " + \
                f"{est_remaining_time_h} h\n" + \
                f"Average episodic reward across all environments: {self._episodic_rewards_env_avrg[self._log_it_counter, :, :].item()}\n" + \
                f"Average episodic rewards across all environments {self._reward_names_str}: {self._episodic_sub_rewards_env_avrg[self._log_it_counter, :]}\n" + \
                f"Current env. step fps: {self._env_step_fps[self._log_it_counter].item()}, time for experience collection {self._collection_dt[self._log_it_counter].item()} s\n" + \
                f"Current env step rt factor: {self._env_step_rt_factor[self._log_it_counter].item()}\n" + \
                f"Current policy update fps: {self._policy_update_fps[self._log_it_counter].item()}, time for policy updates {self._policy_update_dt[self._log_it_counter].item()} s\n"
            Journal.log(self.__class__.__name__,
                "_post_step",
                info,
                LogType.INFO,
                throw_when_excep = True)

    def _init_dbdata(self):

        # initalize some debug data

        # rollout phase
        self._collection_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._collection_t = -1.0
        self._env_step_fps = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._env_step_rt_factor = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._policy_update_t = -1.0
        self._policy_update_dt = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        self._policy_update_fps = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
        
        self._n_of_played_episodes = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_timesteps_done = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._n_policy_updates = torch.full((self._db_data_size, 1), 
                    dtype=torch.int32, fill_value=0, device="cpu")
        self._elapsed_min = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        self._learning_rates = torch.full((self._db_data_size, 2), 
                    dtype=torch.float32, fill_value=0, device="cpu")
        
        # reward db data
        tot_ep_rew_shape = self._episodic_reward_getter.get_rollout_avrg_total_reward().shape
        tot_ep_rew_shape_env_avrg_shape = self._episodic_reward_getter.get_rollout_avrg_total_reward_env_avrg().shape
        rollout_avrg_rew_shape = self._episodic_reward_getter.get_rollout_avrg_reward().shape
        rollout_avrg_rew_env_avrg_shape = self._episodic_reward_getter.get_rollout_reward_env_avrg().shape
        self._reward_names = self._episodic_reward_getter.reward_names()
        self._reward_names_str = "[" + ', '.join(self._reward_names) + "]"
        self._episodic_rewards = torch.full((self._db_data_size, tot_ep_rew_shape[0], tot_ep_rew_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_rewards_env_avrg = torch.full((self._db_data_size, tot_ep_rew_shape_env_avrg_shape[0], tot_ep_rew_shape_env_avrg_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards = torch.full((self._db_data_size, rollout_avrg_rew_shape[0], rollout_avrg_rew_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        self._episodic_sub_rewards_env_avrg = torch.full((self._db_data_size, rollout_avrg_rew_env_avrg_shape[0], rollout_avrg_rew_env_avrg_shape[1]), 
                                        dtype=torch.float32, fill_value=0.0, device="cpu")
        
        # custom data from env
        self._custom_env_data = {}
        db_data_names = list(self._env.custom_db_data.keys())
        for dbdatan in db_data_names:
            self._custom_env_data[dbdatan] = {}
            rollout_stat=self._env.custom_db_data[dbdatan].get_rollout_stat()
            self._custom_env_data[dbdatan]["rollout_stat"] = torch.full((self._db_data_size, 
                                                                rollout_stat.shape[0], 
                                                                rollout_stat.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_env_avrg=self._env.custom_db_data[dbdatan].get_rollout_stat_env_avrg()
            self._custom_env_data[dbdatan]["rollout_stat_env_avrg"] = torch.full((self._db_data_size, 
                                                                        rollout_stat_env_avrg.shape[0], 
                                                                        rollout_stat_env_avrg.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_comp=self._env.custom_db_data[dbdatan].get_rollout_stat_comp()
            self._custom_env_data[dbdatan]["rollout_stat_comp"] = torch.full((self._db_data_size, 
                                                                    rollout_stat_comp.shape[0], 
                                                                    rollout_stat_comp.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")
            rollout_stat_comp_env_avrg=self._env.custom_db_data[dbdatan].get_rollout_stat_comp_env_avrg()
            self._custom_env_data[dbdatan]["rollout_stat_comp_env_avrg"] = torch.full((self._db_data_size, rollout_stat_comp_env_avrg.shape[0], rollout_stat_comp_env_avrg.shape[1]), 
                    dtype=torch.float32, fill_value=0.0, device="cpu")

        # algorithm-specific db info
        self._qf1_vals = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf2_vals = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf1_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf2_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._qf_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._actor_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._alphas = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        self._alpha_loss = torch.full((self._db_data_size, 1), 
                    dtype=torch.float32, fill_value=torch.nan, device="cpu")
        
    def _init_params(self):

        self._dtype = self._env.dtype()

        self._num_envs = self._env.n_envs()
        self._obs_dim = self._env.obs_dim()
        self._actions_dim = self._env.actions_dim()

        self._run_name = "DefaultRun" # default
        self._env_name = self._env.name()
        self._episode_timeout_lb, self._episode_timeout_ub = self._env.episode_timeout_bounds()
        self._task_rand_timeout_lb, self._task_rand_timeout_ub = self._env.task_rand_timeout_bounds()
        self._env_n_action_reps = self._env.n_action_reps()
        
        self._use_gpu = self._env.using_gpu()
        self._torch_device = torch.device("cpu") # defaults to cpu
        self._torch_deterministic = True

        # main algo settings
        self._replay_bf_full = False
        self._warmstart_timesteps = int(5e3)
        self._warmstart_vectimesteps = round(self._warmstart_timesteps/self._num_envs)

        self._replay_buffer_size_nominal = int(1e6) # 32768
        self._replay_buffer_size_vec = self._replay_buffer_size_nominal//self._num_envs # 32768
        self._replay_buffer_size = self._replay_buffer_size_vec*self._num_envs
        self._batch_size = 256
        self._total_timesteps = int(1e6)
        self._total_timesteps = self._total_timesteps//self._env_n_action_reps # correct with n of action reps
        self._total_timesteps_vec = self._total_timesteps // self._num_envs
        self._total_timesteps = self._total_timesteps_vec * self._num_envs # actual n transitions

        self._lr_policy = 3e-4
        self._lr_q = 1e-3

        self._discount_factor = 0.99
        self._smoothing_coeff = 0.005
        self._noise_clip = 0.5

        self._policy_freq = 2
        self._trgt_net_freq = 1

        self._autotune = True
        self._target_entropy = None
        self._log_alpha = None
        self._alpha = 0.2
        self._a_optimizer = None
        
        # debug
        self._m_checkpoint_freq = 5120 # n timesteps after which a checkpoint model is dumped
        self._db_vecstep_frequency = 512 # log db data every n (vectorized) timesteps
        
        self._db_data_size = round(self._total_timesteps_vec/self._db_vecstep_frequency)+self._db_vecstep_frequency
        # write them to hyperparam dictionary for debugging
        self._hyperparameters["n_envs"] = self._num_envs
        self._hyperparameters["obs_dim"] = self._obs_dim
        self._hyperparameters["actions_dim"] = self._actions_dim
        # self._hyperparameters["critic_size"] = self._critic_size
        # self._hyperparameters["actor_size"] = self._actor_size
        self._hyperparameters["seed"] = self._seed
        self._hyperparameters["using_gpu"] = self._use_gpu
        self._hyperparameters["total_timesteps_vec"] = self._total_timesteps_vec
        self._hyperparameters["episodes timeout lb"] = self._episode_timeout_lb
        self._hyperparameters["episodes timeout ub"] = self._episode_timeout_ub
        self._hyperparameters["task rand timeout lb"] = self._task_rand_timeout_lb
        self._hyperparameters["task rand timeout ub"] = self._task_rand_timeout_ub
        
        self._hyperparameters["warmstart_timesteps"] = self._warmstart_timesteps
        self._hyperparameters["_warmstart_vectimesteps"] = self._warmstart_vectimesteps
        self._hyperparameters["_replay_buffer_size_nominal"] = self._replay_buffer_size_nominal
        self._hyperparameters["_batch_size"] = self._batch_size
        self._hyperparameters["_total_timesteps"] = self._total_timesteps
        self._hyperparameters["_lr_policy"] = self._lr_policy
        self._hyperparameters["_lr_q"] = self._lr_q
        self._hyperparameters["_discount_factor"] = self._discount_factor
        self._hyperparameters["_smoothing_coeff"] = self._smoothing_coeff
        self._hyperparameters["_noise_clip"] = self._noise_clip
        self._hyperparameters["_policy_freq"] = self._policy_freq
        self._hyperparameters["_trgt_net_freq"] = self._trgt_net_freq
        self._hyperparameters["_autotune"] = self._autotune
        self._hyperparameters["_target_entropy"] = self._target_entropy
        self._hyperparameters["_log_alpha"] = self._log_alpha
        self._hyperparameters["_alpha"] = self._alpha
        self._hyperparameters["_m_checkpoint_freq"] = self._m_checkpoint_freq
        self._hyperparameters["_db_vecstep_frequency"] = self._db_vecstep_frequency
        self._hyperparameters["_m_checkpoint_freq"] = self._m_checkpoint_freq

        # small debug log
        info = f"\nUsing \n" + \
            f"total (vectorized) timesteps to be simulated {self._total_timesteps_vec}\n" + \
            f"total timesteps to be simulated {self._total_timesteps}\n" + \
            f"warmstart timesteps {self._warmstart_timesteps}\n" + \
            f"replay buffer nominal size {self._replay_buffer_size_nominal}\n" + \
            f"replay buffer size {self._replay_buffer_size}\n" + \
            f"batch size {self._batch_size}\n" + \
            f"policy update freq {self._policy_freq}\n" + \
            f"target networks freq {self._trgt_net_freq}\n" + \
            f"episode timeout max steps {self._episode_timeout_ub}\n" + \
            f"episode timeout min steps {self._episode_timeout_lb}\n" + \
            f"task rand. max n steps {self._task_rand_timeout_ub}\n" + \
            f"task rand. min n steps {self._task_rand_timeout_lb}\n" + \
            f"number of action reps {self._env_n_action_reps}\n"
        Journal.log(self.__class__.__name__,
            "_init_params",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._vec_transition_counter = 0
        self._log_it_counter = 0

    def _init_replay_buffers(self):
        
        self._bpos = 0

        self._obs = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._actions = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._actions_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._rewards = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, 1),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device)
        self._next_obs = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, self._obs_dim),
                        fill_value=torch.nan,
                        dtype=self._dtype,
                        device=self._torch_device) 
        self._next_terminal = torch.full(size=(self._replay_buffer_size_vec, self._num_envs, 1),
                        fill_value=False,
                        dtype=self._dtype,
                        device=self._torch_device)

    def _add_experience(self, 
            obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
            next_obs: torch.Tensor, 
            next_terminal: torch.Tensor) -> None:
        
        self._obs[self._bpos] = obs
        self._next_obs[self._bpos] = next_obs
        self._actions[self._bpos] = actions
        self._rewards[self._bpos] = rewards
        self._next_terminal[self._bpos] = next_terminal

        self._bpos += 1
        if self._bpos == self._replay_buffer_size_vec:
            self._replay_bf_full = True
            self._bpos = 0
    
    def _sample(self):
        
        up_to = self._replay_buffer_size if self._replay_bf_full else self._bpos

        shuffled_buffer_idxs = torch.randint(0, up_to,
                                        (self._batch_size,)) # randomizing 

        batched_obs = self._obs.reshape((-1, self._env.obs_dim()))
        batched_next_obs = self._next_obs.reshape((-1, self._env.obs_dim()))
        batched_actions = self._actions.reshape((-1, self._env.actions_dim()))
        batched_rewards = self._rewards.reshape(-1)
        batched_terminal = self._next_terminal.reshape(-1)

        sampled_obs = batched_obs[shuffled_buffer_idxs]
        sampled_next_obs = batched_next_obs[shuffled_buffer_idxs]
        sampled_actions = batched_actions[shuffled_buffer_idxs]
        sampled_rewards = batched_rewards[shuffled_buffer_idxs]
        sampled_terminal = batched_terminal[shuffled_buffer_idxs]

        return sampled_obs, sampled_actions,\
            sampled_next_obs,\
            sampled_rewards, \
            sampled_terminal

    def _sample_random_actions(self):
        
        actions = self._env.get_actions()

        action_scale = (self._env.get_actions_ub()-self._env.get_actions_lb())/2.0
        action_offset = (self._env.get_actions_ub()+self._env.get_actions_lb())/2.0

        random_actions = torch.randn_like(actions) * action_scale + action_offset

        return random_actions
    
    def _switch_training_mode(self, 
                    train: bool = True):

        self._agent.train(train)

    def _init_algo_shared_data(self,
                static_params: Dict):

        self._shared_algo_data = SharedRLAlgorithmInfo(namespace="CleanPPO",
                is_server=True, 
                static_params=static_params,
                verbose=self._verbose, 
                vlevel=VLevel.V2, 
                safe=False,
                force_reconnection=True)

        self._shared_algo_data.run()
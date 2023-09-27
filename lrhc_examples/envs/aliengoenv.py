from omni_custom_gym.gym.omni_vect_env.vec_envs import RobotVecEnv

from lrhc_examples.controllers.aliengo_rhc.aliengorhc_cluster_client import AliengoRHClusterClient

import torch 
import numpy as np

class AliengoEnv(RobotVecEnv):

    def set_task(self, 
                task, 
                backend="torch", 
                sim_params=None, 
                init_sim=True, 
                np_array_dtype = np.float32, 
                verbose = False, 
                debug = False) -> None:

        super().set_task(task, 
                backend=backend, 
                sim_params=sim_params, 
                init_sim=init_sim)
        
        # now the task and the simulation is guaranteed to be initialized
        # -> we have the data to initialize the cluster client
        self.cluster_client = AliengoRHClusterClient(cluster_size=task.num_envs, 
                        device=task.torch_device, 
                        cluster_dt=task.cluster_dt, 
                        control_dt=task.integration_dt, 
                        jnt_names = task.robot_dof_names, 
                        np_array_dtype = np_array_dtype, 
                        verbose = verbose, 
                        debug = debug)
        
        self.init_cluster_cmd_to_safe_vals()

        self._is_cluster_ready = False
        
    def step(self, 
        index: int, 
        actions = None):
        
        if self.cluster_client.is_first_control_step():
            
            # first time the cluster is ready (i.e. the controllers are ready and connected)

            self.task.init_root_abs_offsets() # we get the current absolute positions and use them as 
            # references

        if self.cluster_client.is_cluster_instant(index):
            
            # assign last robot state observation to the cluster client
            self.update_cluster_state()

            # the control cluster may run at a different rate wrt the simulation

            self.cluster_client.solve() # we solve all the underlying TOs in the cluster
            # (the solve will do nothing unless the cluster is ready)

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]: " + \
                "cluster client solve time -> " + \
                str(self.cluster_client.solution_time))

        if self.cluster_client.cluster_ready() and \
            self.cluster_client.controllers_active:
            
            self.task.pre_physics_step(self.cluster_client.controllers_cmds, 
                        is_first_control_step = self.cluster_client.is_first_control_step())
            
        else:

            self.task.pre_physics_step(None, 
                        is_first_control_step = False)

        self._world.step(render=self._render)
        
        self.sim_frame_count += 1

        observations = self.task.get_observations()

        rewards = self.task.calculate_metrics()
        dones = self.task.is_done()
        info = {}

        return observations, rewards, dones, info
        
    def reset(self):

        self._world.reset()

        self.task.reset()
        
        self._world.step(render=self._render)

        observations = self.task.get_observations()

        self.init_cluster_cmd_to_safe_vals()

        return observations
    
    def update_cluster_state(self):

        self.cluster_client.robot_states.root_state.p[:, :] = torch.sub(self.task.root_p, 
                                                                self.task.root_abs_offsets) # we only get the relative position
        # w.r.t. the initial spawning pose
        self.cluster_client.robot_states.root_state.q[:, :] = self.task.root_q
        self.cluster_client.robot_states.root_state.v[:, :] = self.task.root_v
        self.cluster_client.robot_states.root_state.omega[:, :] = self.task.root_omega
        self.cluster_client.robot_states.jnt_state.q[:, :] = self.task.jnts_q
        self.cluster_client.robot_states.jnt_state.v[:, :] = self.task.jnts_v

    def init_cluster_cmd_to_safe_vals(self):
        
        self.task._jnt_imp_controller.set_refs(pos_ref = self.task._homer.get_homing())
        self.task._jnt_imp_controller.apply_refs()

    def close(self):

        self.cluster_client.close()
        
        super().close() # this has to be called last 
        # so that isaac's simulation is close properly

        
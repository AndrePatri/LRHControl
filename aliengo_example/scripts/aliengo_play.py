import os
script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

import numpy as np
import torch

#from stable_baselines3 import PPO
from aliengo_example.envs.aliengoenv import AliengoEnv 

env = AliengoEnv(headless=False, 
            enable_livestream=False, 
            enable_viewport=False) # create environment

# now we can import the task (not before, since Omni plugins are loaded 
# upon environment initialization)
from aliengo_example.tasks.aliengo_example_task import AliengoExampleTask

num_envs = 9 # 9, 3, 5
sim_params = {}
sim_params["use_gpu_pipeline"] = True
sim_params["integration_dt"] = 1.0/100.0
sim_params["rendering_dt"] = 1.0/50.0
sim_params["substeps"] = 1
sim_params["gravity"] = np.array([0.0, 0.0, -9.81])
sim_params["enable_scene_query_support"] = True
sim_params["replicate_physics"] = True
sim_params["use_flatcache"] = True
sim_params["disable_contact_processing"] = False
if sim_params["use_gpu_pipeline"]:
    sim_params["device"] = "cuda"
else:
    sim_params["device"] = "cpu"

device = sim_params["device"]

control_clust_dt = sim_params["integration_dt"] * 2
integration_dt = sim_params["integration_dt"]


dtype = "float32" # Isaac requires data to be float32, so this should not be touched
if dtype == "float64":
    dtype_np = np.float64 
    dtype_torch = torch.float64
if dtype == "float32":
    dtype_np = np.float32
    dtype_torch = torch.float32

# this has to be the same wrt the cluster server, otherwise
# messages are not read/written properly

task = AliengoExampleTask(cluster_dt = control_clust_dt, 
                        integration_dt = integration_dt,
                        num_envs = num_envs, 
                        cloning_offset = np.array([0.0, 0.0, 1.0]), 
                        device = device, 
                        dtype=dtype_torch, 
                        default_jnt_stiffness=100.0, 
                        default_jnt_damping=10.0) # create task

env.set_task(task, 
        backend="torch", 
        sim_params = sim_params, 
        np_array_dtype = dtype_np, 
        verbose=True, 
        debug=True) # add the task to the environment 
# (includes spawning robots and launching the cluster client for the controllers)

# Run inference on the trained policy
#model = PPO.load("ppo_cartpole")
# env._world.reset()
obs = env.reset()
# env._world.pause()

import time
rt_time_reset = 100
rt_factor = 1.0
real_time = 0.0
sim_time = 0.0
i = 0
start_time = time.perf_counter()
start_time_loop = 0
rt_factor_reset_n = 100 
rt_factor_counter = 0

while env._simulation_app.is_running():

    start_time_loop = time.perf_counter()
    
    if ((i + 1) % rt_factor_reset_n) == 0:

        rt_factor_counter = 0

        start_time = time.perf_counter()

        sim_time = 0

    # action, _states = model.predict(obs)
    
    obs, rewards, dones, info = env.step(index=i) 
    
    now = time.perf_counter()
    real_time = now - start_time
    sim_time += sim_params["integration_dt"]
    rt_factor = sim_time / real_time
    
    i+=1 # updating simulation iteration number
    rt_factor_counter = rt_factor_counter + 1

    print(f"[{script_name}]" + "[info]: current RT factor-> " + str(rt_factor))
    print(f"[{script_name}]" + "[info]: current training RT factor-> " + str(rt_factor * num_envs))
    print(f"[{script_name}]" + "[info]: real_time-> " + str(real_time))
    print(f"[{script_name}]" + "[info]: sim_time-> " + str(sim_time))
    print(f"[{script_name}]" + "[info]: loop execution time-> " + str(now - start_time_loop))

env.close()
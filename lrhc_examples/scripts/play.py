# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of LRhcExamples and distributed under the General Public License version 2 license.
# 
# LRhcExamples is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# LRhcExamples is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with LRhcExamples.  If not, see <http://www.gnu.org/licenses/>.
# 
import os
script_name = os.path.splitext(os.path.basename(os.path.abspath(__file__)))[0]

import numpy as np
import torch

#from stable_baselines3 import PPO
from lrhc_examples.envs.lrhcenv import DummyEnv 

env = DummyEnv(headless=False, 
            enable_livestream=False, 
            enable_viewport=False) # create environment

# now we can import the task (not before, since Omni plugins are loaded 
# upon environment initialization)
from lrhc_examples.tasks.dummy_task import ExampleTask

from omni_robo_gym.utils.shared_sim_info import SharedSimInfo

print_sim_info = False

num_envs = 1 # 9, 3, 5
sim_params = {}
sim_params["use_gpu_pipeline"] = False # if True, data will not be exported back to the CPU RAM 
# any attempt at reading / printing this data for code that is executed in the CPU will return wrong values.
# -> more performant version of the simulation
sim_params["integration_dt"] = 1.0/100.0
sim_params["rendering_dt"] = 1.0/25.0
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

# create task
robot_names = ["aliengo0", "centauro0", "aliengo1", "centauro1"] # robot names
robot_pkg_names = ["aliengo", "centauro", "aliengo", "centauro"] # robot type

# robot_names = ["centauro0"] # robot names
# robot_pkg_names = ["centauro"] # robot type

contact_prims = {} # contact sensors to be added
contact_prims["aliengo0"] = [] # foot contact sensors
contact_prims["centauro0"] = ["wheel_1", "wheel_2", "wheel_3", "wheel_4"]
contact_prims["aliengo1"] =  []
contact_prims["centauro1"] = []

contact_offsets = {}
contact_offsets["centauro0"] = {}
for i in range(0, len(contact_prims["centauro0"])):
    
    contact_offsets["centauro0"][contact_prims["centauro0"][i]] = \
        np.array([0.0, 0.0, 0.0])

contact_offsets["aliengo1"] = {}
contact_offsets["centauro1"] = {}

sensor_radius = {}
sensor_radius["centauro0"] = {}
sensor_radius["centauro1"] = {}
sensor_radius["aliengo0"] = {}
sensor_radius["aliengo1"] = {}
for i in range(0, len(contact_prims["centauro0"])):
    
    sensor_radius["centauro0"][contact_prims["centauro0"][i]] = 0.3

task = ExampleTask(cluster_dt = control_clust_dt, 
            integration_dt = integration_dt,
            num_envs = num_envs, 
            cloning_offset = np.array([[0.0, 0.0, 0.4]] * num_envs), 
            env_spacing=7.0,
            spawning_radius=2.0,
            device = device, 
            dtype=dtype_torch, 
            use_flat_ground = True, 
            default_jnt_stiffness=300.0, 
            default_jnt_damping=10.0, 
            robot_names = robot_names,
            robot_pkg_names = robot_pkg_names,
            contact_prims = contact_prims, 
            contact_offsets = contact_offsets,
            sensor_radius = sensor_radius) 

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

shared_sim_info = SharedSimInfo() # sim. info to be broadcasted
shared_sim_info.start(gpu_pipeline_active=sim_params["use_gpu_pipeline"], 
                    integration_dt=integration_dt,
                    rendering_dt=sim_params["rendering_dt"], 
                    cluster_dt=control_clust_dt)

while env._simulation_app.is_running():
    
    if ((i + 1) % rt_factor_reset_n) == 0:

        rt_factor_counter = 0

        start_time = time.perf_counter()

        sim_time = 0
    
    start_time_step = time.perf_counter()

    obs, rewards, dones, info = env.step(index=i) 
    
    now = time.perf_counter()

    real_time = now - start_time
    sim_time += sim_params["integration_dt"]
    rt_factor = sim_time / real_time
    
    shared_sim_info.update(sim_rt_factor=rt_factor, 
                        cumulative_rt_factor=rt_factor * num_envs, 
                        time_for_sim_stepping=now - start_time_step)
    
    i+=1 # updating simulation iteration number
    rt_factor_counter = rt_factor_counter + 1

    if print_sim_info:

        print(f"[{script_name}]" + "[info]: current RT factor-> " + str(rt_factor))
        print(f"[{script_name}]" + "[info]: current training RT factor-> " + str(rt_factor * num_envs))
        print(f"[{script_name}]" + "[info]: real_time-> " + str(real_time))
        print(f"[{script_name}]" + "[info]: sim_time-> " + str(sim_time))
        print(f"[{script_name}]" + "[info]: loop execution time-> " + str(now - start_time_step))

    contact_report = task.contact_sensors["centauro0"][0][0].get_current_frame() # LF foot

    print("#############")
    print(contact_report)

print("[main][info]: closing environment and simulation")

env.close()
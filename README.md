### AliengoExample package

|Isaac simulation|Debugging GUI|   
|:----------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
|  <img src="aliengo_example/docs/images/isaac_sim.png" alt="drawing" width="600"/> | <img src="aliengo_example/docs/images/gui_example.png" alt="drawing" width="500"/>

Example scripts and implementations using the following two packages:
 
- [ControlClusterUtils](https://github.com/AndPatr/ControlClusterUtils): utilities to create a CPU-based controllers cluster to be interfaced with GPU-based simulators, also available through Anaconda [here](https://anaconda.org/AndrePatri/control_cluster_utils).
- [OmniCustomGym](https://github.com/AndPatr/OmniCustomGym): custom implementations of Tasks and Gyms for for Omniverse Isaac Sim based on Gymnasium. Easy URDF and SRDF import/cloning and simulation configuration exploiting Omniverse API. This package is also available through Anaconda [here](https://anaconda.org/AndrePatri/omni_custom_gym).

Installation instructions:

- The preferred way of using AliengoExample package is to employ the provided environment at [OmniCustomGym](https://github.com/AndPatr/OmniCustomGym). Follow the installation instruction to setup the environment and install Isaac Sim.

- From the root folder install the package in editable mode with ```pip install --no-deps -e .```

- Clone and install [unitree_ros](https://github.com/AndrePatri/unitree_ros). Note: install in your workspace only the ```aliengo_description``` subpackage. 


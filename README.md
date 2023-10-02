### LRhcExamples package

|Isaac simulation|Debugging GUI|   
|:----------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
|  <img src="lrhc_examples/docs/images/multirobot_support.png" alt="drawing" width="600" /> | <img src="lrhc_examples/docs/images/gui_light.png#gh-dark-mode-onlylrhc_examples/docs/images/gui_light.png#gh-light-mode-only" alt="drawing" width="500"/>


Learning-based Receding Horizon Control examples on [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html), built exploiting the following two packages:
 
- [ControlClusterUtils](https://github.com/AndPatr/ControlClusterUtils): utilities to create an interface between a CPU-based control cluster and GPU-based simulators (also available through Anaconda [here](https://anaconda.org/AndrePatri/control_cluster_utils)).
- [OmniCustomGym](https://github.com/AndPatr/OmniCustomGym): custom implementations of Tasks and Gyms for Robotics exploiting [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html) and [Gymnasium](https://gymnasium.farama.org/). Easy URDF and SRDF importing/cloning, multi-robot support and simulation setup built on top of Isaac's Python API. Also available through Anaconda [here](https://anaconda.org/AndrePatri/omni_custom_gym).

Installation instructions:

- The preferred way of using LRhcExamples package is to employ the provided environment at [OmniCustomGym](https://github.com/AndPatr/OmniCustomGym). Follow the installation instruction to setup the environment and install Omniverse Isaac Sim.

- From the root folder install the package in editable mode with ```pip install --no-deps -e .```

- For using the examples, you need some robot descriptions. In particular: 
    - To be able to use Unitree's Aliengo, clone and install [this](https://github.com/AndrePatri/unitree_ros) fork of *unitree_ros* (note: install in your workspace only the ```aliengo_description``` subpackage).
    - To be able to use Centauro, clone and install [this](https://github.com/ADVRHumanoids/iit-centauro-ros-pkg/tree/big_wheels_v2.10) repo.
    


### LRhcExamples package

|Isaac simulation|Debugging GUI|   
|:----------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
|  <img src="lrhc_examples/docs/images/isaac_sim.png" alt="drawing" width="600" media="(prefers-color-scheme: light)"/> <img src="lrhc_examples/docs/images/isaac_sim.png" alt="drawing" width="600" media="(prefers-color-scheme: dark)"/> | <img src="lrhc_examples/docs/images/gui_light.png#gh-dark-mode-onlylrhc_examples/docs/images/gui_light.png#gh-light-mode-only" alt="drawing" width="500"/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/25423296/163456776-7f95b81a-f1ed-45f7-b7ab-8fa810d529fa.png">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
</picture>

Learning-based Receding Horizon Control examples on [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html), built exploiting the following two packages:
 
- [ControlClusterUtils](https://github.com/AndPatr/ControlClusterUtils): utilities to create an interface between a CPU-based control cluster and GPU-based simulators (also available through Anaconda [here](https://anaconda.org/AndrePatri/control_cluster_utils)).
- [OmniCustomGym](https://github.com/AndPatr/OmniCustomGym): custom implementations of Tasks and Gyms for Robotic exploiting [Omniverse Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim.html) and [Gymnasium](https://gymnasium.farama.org/). Easy URDF and SRDF importing/cloning and simulation setup built on top of Isaac's Python API. Also available through Anaconda [here](https://anaconda.org/AndrePatri/omni_custom_gym).

Installation instructions:

- The preferred way of using LRhcExamples package is to employ the provided environment at [OmniCustomGym](https://github.com/AndPatr/OmniCustomGym). Follow the installation instruction to setup the environment and install Omniverse Isaac Sim.

- From the root folder install the package in editable mode with ```pip install --no-deps -e .```

- For using the examples, you need some robot description resources. In particular: 
    - To be able to use Unitree's Aliengo, clone and install [this](https://github.com/AndrePatri/unitree_ros) fork of *unitree_ros* (note: install in your workspace only the ```aliengo_description``` subpackage).
    


from PyQt5.QtWidgets import QWidget

from control_cluster_bridge.utilities.debugger_gui.gui_exts import SharedDataWindow
from control_cluster_bridge.utilities.debugger_gui.plot_utils import RtPlotWindow
from control_cluster_bridge.utilities.debugger_gui.plot_utils import WidgetUtils

from SharsorIPCpp.PySharsorIPC import VLevel

from lrhc_control.utils.shared_data.training_env import SharedTrainingEnvInfo
from control_cluster_bridge.utilities.shared_data.rhc_data import RobotState
from lrhc_control.utils.shared_data.agent_refs import AgentRefs
from lrhc_control.utils.shared_data.training_env import Observations
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Rewards
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Truncations
from lrhc_control.utils.shared_data.training_env import Terminations
from lrhc_control.utils.shared_data.training_env import TimeUnlimitedTasksEpCounter
from lrhc_control.utils.shared_data.algo_infos import SharedRLAlgorithmInfo

import numpy as np

class TrainingEnvData(SharedDataWindow):

    def __init__(self, 
        update_data_dt: int,
        update_plot_dt: int,
        window_duration: int,
        window_buffer_factor: int = 2,
        namespace = "",
        parent: QWidget = None, 
        verbose = False,
        add_settings_tab = False,
        ):
        
        name = "TrainingEnvData"

        self.current_cluster_index = 0

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 1,
            grid_n_cols = 1,
            namespace = namespace,
            name = name,
            parent = parent, 
            verbose = verbose,
            add_settings_tab = add_settings_tab,
            )

    def _init_shared_data(self):
        
        is_server = False
        
        self.shared_data_clients.append(SharedTrainingEnvInfo(is_server=is_server,
                                            namespace=self.namespace, 
                                            verbose=True, 
                                            vlevel=VLevel.V2))
        
        self.shared_data_clients.append(RobotState(namespace=self.namespace,
                                            is_server=False, 
                                            with_gpu_mirror=False,
                                            safe=False,
                                            verbose=True,
                                            vlevel=VLevel.V2))

        self.shared_data_clients.append(AgentRefs(namespace=self.namespace,
                                is_server=False,
                                with_gpu_mirror=False,
                                safe=False,
                                verbose=True,
                                vlevel=VLevel.V2))
        
        self.shared_data_clients.append(Observations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(Actions(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))

        self.shared_data_clients.append(Rewards(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(TotRewards(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(Truncations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))

        self.shared_data_clients.append(Terminations(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(TimeUnlimitedTasksEpCounter(namespace=self.namespace,
                                            is_server=False,
                                            verbose=True,
                                            vlevel=VLevel.V2,
                                            safe=False,
                                            with_gpu_mirror=False))
        
        self.shared_data_clients.append(SharedRLAlgorithmInfo(is_server=is_server,
                                            namespace="CleanPPO", 
                                            verbose=True, 
                                            vlevel=VLevel.V2))

        for client in self.shared_data_clients:

            client.run()

    def _post_shared_init(self):
        
        self.grid_n_rows = 5

        self.grid_n_cols = 2

    def _initialize(self):
        
        cluster_size = self.shared_data_clients[1].n_robots()

        cluster_idx_legend = [""] * cluster_size
        for i in range(cluster_size):
            cluster_idx_legend[i] = str(i)
        
        obs_legend = [""] * 4
        for i in range(len(obs_legend)):
            obs_legend[i] = str(i)

        reward_legend = [""] * 3
        for i in range(len(reward_legend)):
            reward_legend[i] = str(i)

        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[0].param_keys),
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Training env. info", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[0].param_keys, 
                                    ylabel=""))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[10].param_keys),
                                    n_data = 1,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"RL algorithm info", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[10].param_keys, 
                                    ylabel=""))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[3].n_cols,
                                    n_data = cluster_size,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Observations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[3].col_names(), 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=self.shared_data_clients[4].n_cols,
                                    n_data = cluster_size,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Actions", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[4].col_names(), 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[5].col_names()),
                                    n_data = cluster_size,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rewards - detailed", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[5].col_names(), 
                                    ylabel="[float]"))

        self.rt_plotters.append(RtPlotWindow(data_dim=len(self.shared_data_clients[6].col_names()),
                                    n_data = cluster_size,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Rewards", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=self.shared_data_clients[6].col_names(), 
                                    ylabel="[float]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                                    n_data = cluster_size,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Truncations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=[""], 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                                    n_data = cluster_size,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Terminations", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=[""], 
                                    ylabel="[bool]"))
        
        self.rt_plotters.append(RtPlotWindow(data_dim=1,
                                    n_data = cluster_size,
                                    update_data_dt=self.update_data_dt, 
                                    update_plot_dt=self.update_plot_dt,
                                    window_duration=self.window_duration, 
                                    parent=None, 
                                    base_name=f"Episode counter", 
                                    window_buffer_factor=self.window_buffer_factor, 
                                    legend_list=[""], 
                                    ylabel="[int]"))
        
        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 1)
        self.grid.addFrame(self.rt_plotters[4].base_frame, 2, 0)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 2, 1)
        self.grid.addFrame(self.rt_plotters[6].base_frame, 3, 0)
        self.grid.addFrame(self.rt_plotters[7].base_frame, 3, 1)
        self.grid.addFrame(self.rt_plotters[8].base_frame, 4, 0)

    def _finalize_grid(self):
                
        widget_utils = WidgetUtils()

        settings_frames = []

        cluster_size = self.shared_data_clients[1].n_robots()

        node_index_slider = widget_utils.generate_complex_slider(
                        parent=None, 
                        parent_layout=None,
                        min_shown=f"{0}", min= 0, 
                        max_shown=f"{cluster_size - 1}", 
                        max=cluster_size - 1, 
                        init_val_shown=f"{0}", init=0, 
                        title="cluster index", 
                        callback=self._update_cluster_idx)
        
        settings_frames.append(node_index_slider)
        
        self.grid.addToSettings(settings_frames)

    def _update_cluster_idx(self,
                    idx: int):

        self.current_cluster_index = idx

        self.grid.settings_widget_list[0].current_val.setText(f'{idx}')

        for i in range(2, len(self.rt_plotters)):
            # only switching plots which have n_cluster dim
            self.rt_plotters[i].rt_plot_widget.switch_to_data(data_idx = self.current_cluster_index)

    def update(self,
            index: int):

        # index not used here (no dependency on cluster index)

        if not self._terminated:
            
            # read data on shared memory
            env_data = self.shared_data_clients[0].get().flatten()
            algo_data = self.shared_data_clients[10].get().flatten()

            self.shared_data_clients[1].synch_from_shared_mem()
            self.shared_data_clients[2].rob_refs.root_state.synch_all(read=True, wait=True)
            self.shared_data_clients[3].synch_all(read=True, wait=True) # observations
            self.shared_data_clients[4].synch_all(read=True, wait=True)
            self.shared_data_clients[5].synch_all(read=True, wait=True)
            self.shared_data_clients[6].synch_all(read=True, wait=True)

            self.shared_data_clients[7].synch_all(read=True, wait=True)
            self.shared_data_clients[8].synch_all(read=True, wait=True)
            self.shared_data_clients[9].counter().synch_all(read=True, wait=True)

            self.rt_plotters[0].rt_plot_widget.update(env_data)
            self.rt_plotters[1].rt_plot_widget.update(algo_data)
            self.rt_plotters[2].rt_plot_widget.update(np.transpose(self.shared_data_clients[3].get_numpy_view()))
            self.rt_plotters[3].rt_plot_widget.update(np.transpose(self.shared_data_clients[4].get_numpy_view()))
            self.rt_plotters[4].rt_plot_widget.update(np.transpose(self.shared_data_clients[5].get_numpy_view()))
            self.rt_plotters[5].rt_plot_widget.update(np.transpose(self.shared_data_clients[6].get_numpy_view()))

            self.rt_plotters[6].rt_plot_widget.update(np.transpose(self.shared_data_clients[7].get_numpy_view()))
            self.rt_plotters[7].rt_plot_widget.update(np.transpose(self.shared_data_clients[8].get_numpy_view()))
            self.rt_plotters[8].rt_plot_widget.update(np.transpose(self.shared_data_clients[9].counter().get_numpy_view()))


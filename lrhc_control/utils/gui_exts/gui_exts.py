from PyQt5.QtWidgets import QWidget

from control_cluster_bridge.utilities.debugger_gui.plot_utils import GridFrameWidget
from control_cluster_bridge.utilities.debugger_gui.plot_utils import RtPlotWindow

from abc import abstractmethod

from typing import TypeVar

class SharedDataWindow():

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            grid_n_rows: int = -1,
            grid_n_cols: int = -1,
            window_buffer_factor: int = 2,
            namespace = "",
            name = "SharedDataWindow",
            parent: QWidget = None, 
            verbose = False,
            add_settings_tab = False, 
            settings_title = "SharedDataSettings"):

        self.add_settings_tab = add_settings_tab

        self.settings_title = settings_title

        self.grid_n_rows = grid_n_rows
        self.grid_n_cols = grid_n_cols

        self.namespace = namespace

        self.name = name

        self.parent = parent

        self.update_data_dt = update_data_dt
        self.update_plot_dt = update_plot_dt
        self.window_duration = window_duration
        self.window_buffer_factor = window_buffer_factor

        self.verbose = verbose

        self.base_frame = None

        self._reset()
    
    def __del__(self):

        self.terminate()
        
    @abstractmethod
    def _initialize(self):
        
        pass

    @abstractmethod
    def _post_shared_init(self):
        
        # things to be done between shared data init and before ui initialization

        pass
    
    @abstractmethod
    def _init_shared_data(self):

        pass
    
    @abstractmethod
    def update(self,
            index: int = 0):

        # index -> index of the shared data client to be used for updating 
        # the plots 
        
        # this would tipically update each plotter in self.rt_plotters
        # using data from shared memory

        pass 
    
    def run(self):
        
        self._reset()

        self._init_shared_data()
        
        self._post_shared_init()

        self._init_ui()

        self._finalize_grid()

        self.grid.finalize()

    def _finalize_grid(self):

        # to be overridden

        pass

    def _reset(self):

        self._terminated = False

        self.cluster_idx = 0

        self.shared_data_clients = []
        
        self.rt_plotters = []
        
    def _init_ui(self):

        self.grid = GridFrameWidget(self.grid_n_rows, self.grid_n_cols, 
                        parent=self.parent,
                        add_settings_tab = self.add_settings_tab, 
                        settings_title=self.settings_title)
        
        self.base_frame = self.grid.base_frame

        self.rt_plotters = []

        self._initialize()

    def swith_pause(self):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.paused = \
                    not self.rt_plotters[i].rt_plot_widget.paused

    def change_sample_update_dt(self, 
                dt: float):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.update_data_sample_dt(dt)

                self.rt_plotters[i].settings_widget.synch_max_window_size()

    def change_plot_update_dt(self, 
                    dt: float):
        
        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.set_timer_interval(dt)
                
    def nightshift(self):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.nightshift()
        
    def dayshift(self):

        if not self._terminated:
            
            for i in range(len(self.rt_plotters)):

                self.rt_plotters[i].rt_plot_widget.dayshift()

    def terminate(self):
        
        for i in range(0, len(self.shared_data_clients)):
                        
            self.shared_data_clients[i].close()
        
        self._terminated = True

SharedDataWindowChild = TypeVar('SharedDataWindowChild', bound='SharedDataWindow')

# Example of extension

from control_cluster_bridge.utilities.shared_data.jnt_imp_control import JntImpCntrlData

from SharsorIPCpp.PySharsorIPC import VLevel

import numpy as np

class JntImpMonitor(SharedDataWindow):

    def __init__(self, 
            update_data_dt: int,
            update_plot_dt: int,
            window_duration: int,
            window_buffer_factor: int = 2,
            namespace = "",
            parent: QWidget = None, 
            verbose = False):

        self.n_jnts = -1
        self.n_envs = -1
        self.jnt_names = []

        super().__init__(update_data_dt = update_data_dt,
            update_plot_dt = update_plot_dt,
            window_duration = window_duration,
            window_buffer_factor = window_buffer_factor,
            grid_n_rows = 3,
            grid_n_cols = 3,
            namespace = namespace,
            name = "JntImpMonitor",
            parent = parent, 
            verbose = verbose)

    def _initialize(self):
        
        self.rt_plotters.append(RtPlotWindow(data_dim = 2 * self.n_jnts,
                    n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Pos VS Pos Ref.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names + \
                                [item + "_ref" for item in self.jnt_names]))
        
        self.rt_plotters.append(RtPlotWindow(data_dim = 2 * self.n_jnts,
                    n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Vel VS Vel Ref.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names + \
                                [item + "_ref" for item in self.jnt_names]))

        self.rt_plotters.append(RtPlotWindow(data_dim = 2 * self.n_jnts,
                    n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Meas. eff VS Imp Eff.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names + \
                                [item + "_imp" for item in self.jnt_names]))

        self.rt_plotters.append(RtPlotWindow(data_dim = self.n_jnts,
                    n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Pos Gains", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names))

        self.rt_plotters.append(RtPlotWindow(data_dim = self.n_jnts,
                    n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Vel Gains", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names))

        self.rt_plotters.append(RtPlotWindow(data_dim = self.n_jnts,
                    n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Pos Err.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names))

        self.rt_plotters.append(RtPlotWindow(data_dim = self.n_jnts,
                    n_data=1, 
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Vel Err.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names))

        self.rt_plotters.append(RtPlotWindow(data_dim = self.n_jnts,
                    n_data=1,  
                    update_data_dt=self.update_data_dt, 
                    update_plot_dt=self.update_plot_dt,
                    window_duration=self.window_duration, 
                    parent=None, 
                    base_name="Eff Feedfor.", 
                    window_buffer_factor=self.window_buffer_factor, 
                    legend_list=self.jnt_names))

        self.grid.addFrame(self.rt_plotters[0].base_frame, 0, 0)
        self.grid.addFrame(self.rt_plotters[1].base_frame, 0, 1)
        self.grid.addFrame(self.rt_plotters[2].base_frame, 0, 2)
        self.grid.addFrame(self.rt_plotters[3].base_frame, 1, 0)
        self.grid.addFrame(self.rt_plotters[4].base_frame, 1, 1)
        self.grid.addFrame(self.rt_plotters[5].base_frame, 2, 0)
        self.grid.addFrame(self.rt_plotters[6].base_frame, 2, 1)
        self.grid.addFrame(self.rt_plotters[7].base_frame, 2, 2)

    def _init_shared_data(self):

        self.shared_data_clients.append(JntImpCntrlData(is_server = False, 
                                            namespace = self.namespace, 
                                            verbose = True, 
                                            vlevel = VLevel.V2,
                                            safe=False))
        
        self.shared_data_clients[0].run()

        self.n_jnts = self.shared_data_clients[0].n_jnts
        self.n_envs = self.shared_data_clients[0].n_envs
        self.jnt_names = self.shared_data_clients[0].jnt_names

    def _post_shared_init(self):

        pass

    def update(self,
            index: int):

        if not self._terminated:
            
            # pos VS pos ref
            self.shared_data_clients[0].pos_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].pos_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].pos_view.n_cols) 
                                                       
            self.shared_data_clients[0].pos_ref_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].pos_ref_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].pos_ref_view.n_cols)

            pos_vs_pos_ref = np.concatenate((self.shared_data_clients[0].pos_view.numpy_view[index, :],
                                            self.shared_data_clients[0].pos_ref_view.numpy_view[index, :]), 
                                            axis=0) 

            self.rt_plotters[0].rt_plot_widget.update(pos_vs_pos_ref.flatten())

            # vel VS vel ref
            self.shared_data_clients[0].vel_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].vel_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].vel_view.n_cols) # synch data
            self.shared_data_clients[0].vel_ref_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].vel_ref_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].vel_ref_view.n_cols) # synch data

            vel_vs_vel_ref = np.concatenate((self.shared_data_clients[0].vel_view.numpy_view[index, :],
                                            self.shared_data_clients[0].vel_ref_view.numpy_view[index, :]), 
                                            axis=0) 

            self.rt_plotters[1].rt_plot_widget.update(vel_vs_vel_ref.flatten())

            # meas. eff VS imp. effort
            self.shared_data_clients[0].eff_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].eff_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].eff_view.n_cols) # synch data
            self.shared_data_clients[0].imp_eff_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].imp_eff_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].imp_eff_view.n_cols) # synch data

            eff_vs_imp_eff = np.concatenate((self.shared_data_clients[0].eff_view.numpy_view[index, :],
                                            self.shared_data_clients[0].imp_eff_view.numpy_view[index, :]), 
                                            axis=0) 

            self.rt_plotters[2].rt_plot_widget.update(eff_vs_imp_eff.flatten())

            # pos gains
            self.shared_data_clients[0].pos_gains_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].pos_gains_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].pos_gains_view.n_cols) # synch data
            self.rt_plotters[3].rt_plot_widget.update(self.shared_data_clients[0].pos_gains_view.numpy_view[index, :].flatten())

            # vel gains
            self.shared_data_clients[0].vel_gains_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].vel_gains_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].vel_gains_view.n_cols) # synch data
            self.rt_plotters[4].rt_plot_widget.update(self.shared_data_clients[0].vel_gains_view.numpy_view[index, :].flatten())

            # pos error
            self.shared_data_clients[0].pos_err_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].pos_err_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].pos_err_view.n_cols) # synch data
            self.rt_plotters[5].rt_plot_widget.update(self.shared_data_clients[0].pos_err_view.numpy_view[index, :].flatten())

            # vel error
            self.shared_data_clients[0].vel_err_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].vel_err_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].vel_err_view.n_cols) # synch data
            self.rt_plotters[6].rt_plot_widget.update(self.shared_data_clients[0].vel_err_view.numpy_view[index, :].flatten())

            # eff. feedforward  
            self.shared_data_clients[0].eff_ff_view.synch(read=True, row_index = 0, col_index = 0, 
                                            n_rows = self.shared_data_clients[0].eff_ff_view.n_rows, 
                                            n_cols = self.shared_data_clients[0].eff_ff_view.n_cols) # synch data
            self.rt_plotters[7].rt_plot_widget.update(self.shared_data_clients[0].eff_ff_view.numpy_view[index, :].flatten())

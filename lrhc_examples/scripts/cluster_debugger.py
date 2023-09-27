from control_cluster_utils.utilities.debugger_gui.cluster_debugger import RtClusterDebugger

if __name__ == "__main__":  

    data_update_dt = 0.01
    plot_update_dt = 0.1

    window_length = 10.0
    window_buffer_factor = 2

    main_window = RtClusterDebugger(data_update_dt=data_update_dt,
                            plot_update_dt=plot_update_dt,
                            window_length=window_length, 
                            window_buffer_factor=window_buffer_factor, 
                            verbose=True)

    main_window.run()

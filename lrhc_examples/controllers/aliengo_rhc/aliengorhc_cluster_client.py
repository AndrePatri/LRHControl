from control_cluster_utils.cluster_client.control_cluster_client import ControlClusterClient

class AliengoRHClusterClient(ControlClusterClient):

    def __init__(self, 
            cluster_size, 
            control_dt,
            cluster_dt,
            jnt_names,
            device, 
            np_array_dtype, 
            verbose, 
            debug):

        self.robot_name = "aliengo"
                
        super().__init__(cluster_size= cluster_size, 
                        control_dt=control_dt, 
                        cluster_dt=cluster_dt, 
                        jnt_names=jnt_names, 
                        device=device, 
                        np_array_dtype=np_array_dtype, 
                        verbose=verbose, 
                        debug=debug, 
                        namespace=self.robot_name)
    
    pass
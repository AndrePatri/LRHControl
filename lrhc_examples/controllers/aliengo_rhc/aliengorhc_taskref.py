from control_cluster_utils.utilities.rhc_defs import RhcTaskRefs

class AliengoRhcTaskRef(RhcTaskRefs):

    def __init__(self,
                n_contacts, 
                index, 
                q_remapping, 
                dtype, 
                verbose, 
                namespace = "aliengo"):
                
        super().__init__(n_contacts=n_contacts, 
                index=index, 
                q_remapping=q_remapping, 
                dtype=dtype, 
                verbose=verbose, 
                namespace=namespace)
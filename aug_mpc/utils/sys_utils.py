import os

class PathsGetter:

    def __init__(self):
        
        self.PACKAGE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
        
        self.CONFIGPATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                    'controllers',
                                    'rhc',
                                    'horizon_based',
                                    'config', 
                                    'rhc_horizon_config.yaml')

        self.RHCDIR = os.path.join(self.PACKAGE_ROOT_DIR, 
                                    'controllers',
                                    'rhc')
        
        self.REMOTENVPATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                    'envs',
                                    'world_interface_base.py')
        
        self.SCRIPTSPATHS = [os.path.join(self.PACKAGE_ROOT_DIR, 
                                    'scripts', 
                                    'launch_world_interface.py'),
                            os.path.join(self.PACKAGE_ROOT_DIR, 
                                    'scripts', 
                                    'launch_train_env.py'),
                            os.path.join(self.PACKAGE_ROOT_DIR, 
                                    'scripts', 
                                    'launch_control_cluster.py')]
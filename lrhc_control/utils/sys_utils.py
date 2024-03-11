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
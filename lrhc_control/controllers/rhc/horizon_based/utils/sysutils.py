import os

class PathsGetter:

    def __init__(self):
        
        self.CONTROLLER_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

        self.CONFIGPATH = os.path.join(self.CONTROLLER_ROOT_DIR, 
                                        'config', 
                                        'rhc_horizon_config.yaml')

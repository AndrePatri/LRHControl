import os

class PathsGetter:

    def __init__(self):
        
        self.PACKAGE_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

        self.DEFAULT_RVIZ_CONFIG_PATH = os.path.join(self.PACKAGE_ROOT_DIR, 
                                            'cfg', 
                                            'config.rviz')

from pynput import keyboard

from lrhc_control.utils.shared_data.agent_refs import AgentRefs

from SharsorIPCpp.PySharsor.wrappers.shared_data_view import SharedTWrapper
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType
from SharsorIPCpp.PySharsorIPC import dtype

import math

import numpy as np

import argparse

class Prova:

    def __init__(self):

        a=1
                
    def _on_press(self, key):

        if hasattr(key, 'char'):
            print("AAAAAA")

    def _on_release(self, key):
        
        if hasattr(key, 'char'):
            print("UUUUUUUU")

    def run(self):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        with keyboard.Listener(on_press=self._on_press, 
                               on_release=self._on_release) as listener:

            listener.join()

if __name__ == "__main__":  

    aaa = Prova()
    aaa.run()
    

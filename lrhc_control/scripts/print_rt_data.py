from lrhc_control.utils.shared_data.training_env import Observations, NextObservations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations
from lrhc_control.utils.shared_data.training_env import Truncations

import time 
from perf_sleep.pyperfsleep import PerfSleep

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal
from SharsorIPCpp.PySharsorIPC import dtype as sharsor_dtype

import torch

if __name__ == "__main__":  

    import argparse
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    
    parser.add_argument('--update_dt', type=float, help='[s]', default=0.1)
    parser.add_argument('--run_for_nominal', type=float, help='[s]]', default=10.0)
    parser.add_argument('--ns', type=str, help='', default="")
    parser.add_argument('--env_idx', type=int, help='', default=0)
    parser.add_argument('--dtype', type=str, help='', default="float")

    args = parser.parse_args()

    update_dt=args.update_dt
    run_for=args.run_for_nominal
    namespace=args.ns
    idx=args.env_idx
    elapsed_tot_nom = 0
    
    dtype=sharsor_dtype.Float
    if args.dtype == "double":
        dtype=sharsor_dtype.Double

    obs = Observations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False,dtype=dtype)
    act = Actions(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False,dtype=dtype)
    rew = TotRewards(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False,dtype=dtype)
    trunc = Truncations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False)
    term = Terminations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False)
    obs.run()
    act.run()
    rew.run()
    trunc.run()
    term.run()

    torch.set_printoptions(precision=2,sci_mode=False,linewidth=50)

    while True:
        try:
            start_time = time.perf_counter() 
            
            print(f"########################")
            print(f"{round(elapsed_tot_nom, 2)} [s] -->\n")
            # read data
            obs.synch_all(read=True, retry=True)
            act.synch_all(read=True, retry=True)
            rew.synch_all(read=True, retry=True)
            trunc.synch_all(read=True, retry=True)
            term.synch_all(read=True, retry=True)

            print("observations:")
            print(obs.get_torch_mirror(gpu=False)[idx, :])
            print("\nactions:")
            print(act.get_torch_mirror(gpu=False)[idx, :])
            print("\nrewards:")
            print(rew.get_torch_mirror(gpu=False)[idx, :])
            print("\nterminations:")
            print(term.get_torch_mirror(gpu=False)[idx, :])
            print("\ntruncations:")
            print(trunc.get_torch_mirror(gpu=False)[idx, :])

            elapsed_time = time.perf_counter() - start_time
            time_to_sleep_ns = int((update_dt - elapsed_time) * 1e+9) # [ns]
            if time_to_sleep_ns < 0:
                warning = f": Could not match desired update dt of {update_dt} s. " + \
                    f"Elapsed time to update {elapsed_time}."
                Journal.log("print_rt_data",
                    "run",
                    warning,
                    LogType.WARN,
                    throw_when_excep = True)
            else:
                PerfSleep.thread_sleep(time_to_sleep_ns) 
            elapsed_tot_nom+=update_dt
            if elapsed_tot_nom>=run_for:
                break
        except KeyboardInterrupt:
            break
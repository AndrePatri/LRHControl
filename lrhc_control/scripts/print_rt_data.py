from lrhc_control.utils.shared_data.training_env import Observations, NextObservations
from lrhc_control.utils.shared_data.training_env import TotRewards
from lrhc_control.utils.shared_data.training_env import Rewards
from lrhc_control.utils.shared_data.training_env import Actions
from lrhc_control.utils.shared_data.training_env import Terminations
from lrhc_control.utils.shared_data.training_env import Truncations
from lrhc_control.utils.shared_data.training_env import EpisodesCounter, TaskRandCounter

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
    parser.add_argument('--env_range', type=int, help='', default=1)
    parser.add_argument('--dtype', type=str, help='', default="float")
    parser.add_argument('--with_counters', action=argparse.BooleanOptionalAction, default=False, help='')

    args = parser.parse_args()

    update_dt=args.update_dt
    run_for=args.run_for_nominal
    namespace=args.ns
    idx=args.env_idx
    env_range=args.env_range
    elapsed_tot_nom = 0
    with_counters = args.with_counters

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
    sub_rew = Rewards(namespace=namespace,
                    is_server=False,
                    verbose=True,
                    vlevel=VLevel.V2,
                    with_gpu_mirror=False)
    trunc = Truncations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False)
    term = Terminations(namespace=namespace,is_server=False,verbose=True, 
                vlevel=VLevel.V2,safe=False,
                with_gpu_mirror=False)
                
    obs.run()
    act.run()
    rew.run()
    sub_rew.run()
    sub_rew_names = sub_rew.col_names()
    trunc.run()
    term.run()
    
    ep_counter = None
    task_counter = None
    if with_counters:
        ep_counter=EpisodesCounter(namespace=namespace,
                    is_server=False,
                    verbose=True,
                    vlevel=VLevel.V2,
                    with_gpu_mirror=False)
        task_counter=TaskRandCounter(namespace=namespace,
                    is_server=False,
                    verbose=True,
                    vlevel=VLevel.V2,
                    with_gpu_mirror=False)
        ep_counter.run()
        task_counter.run()

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
            sub_rew.synch_all(read=True, retry=True)
            trunc.synch_all(read=True, retry=True)
            term.synch_all(read=True, retry=True)

            print("observations:")
            print(obs.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\nactions:")
            print(act.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\nrewards:")
            print(rew.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\nsub-rewards:")
            print(*sub_rew_names, sep = ", ") 
            print(sub_rew.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\nterminations:")
            print(term.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            print("\ntruncations:")
            print(trunc.get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if ep_counter is not None:
                ep_counter.counter().synch_all(read=True, retry=True)
                print("\nep. counter:")
                print(ep_counter.counter().get_torch_mirror(gpu=False)[idx:idx+env_range, :])
            if task_counter is not None:
                task_counter.counter().synch_all(read=True, retry=True)
                print("\ntask counter:")
                print(task_counter.counter().get_torch_mirror(gpu=False)[idx:idx+env_range, :])

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
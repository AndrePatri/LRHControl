from aug_mpc.utils.determinism import deterministic_run
from aug_mpc.utils.custom_arg_parsing import generate_custom_arg_dict

from mpc_hive.utilities.shared_data.sim_data import SharedEnvInfo
from mpc_hive.utilities.shared_data.cluster_data import SharedClusterInfo

from EigenIPC.PyEigenIPC import VLevel, Journal, LogType
from EigenIPC.PyEigenIPC import StringTensorServer
from mpc_hive.utilities.timing import high_resolution_sleep_ns

import os, argparse, sys, types, inspect

import importlib.util
import torch
import signal

algo = None  # global to make it accessible by signal handler
exit_request=False
dummy_step_exit_req=False

def handle_sigint(signum, frame):
    global exit_request, dummy_step_exit_req
    Journal.log("launch_train_env.py",
        "",
        f"Received sigint. Will stop training.",
        LogType.WARN)
    exit_request=True
    dummy_step_exit_req=True # in case dummy_step_loop was used
    
# Function to dynamically import a module from a specific file path
# def import_env_module(env_path):
#     spec = importlib.util.spec_from_file_location("env_module", env_path)
#     env_module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(env_module)
#     return env_module

def import_env_module(env_path, local_env_root: str = None):
    """
    env_path: full path to the child env .py file to exec
    local_env_root: directory where local copies of aug_mpc_envs.training_envs modules live
    """
    if local_env_root is not None:
        local_env_root = os.path.abspath(local_env_root)
        # Make aug_mpc_envs.training_envs look in the bundle dir FIRST, but keep the installed package
        # path as a fallback: bundle-snapshotted env modules take priority, while any dependency that
        # was not snapshotted (e.g. task_reference_utils) still resolves from the installed package
        # instead of raising ModuleNotFoundError. importlib.import_module ensures the real package (with
        # its installed __path__) exists before we prepend the bundle dir.
        pkg_name = "aug_mpc_envs.training_envs"
        try:
            pkg = importlib.import_module(pkg_name)
            installed_path = list(getattr(pkg, "__path__", []))
        except Exception:
            pkg = sys.modules.get(pkg_name, None)
            installed_path = list(getattr(pkg, "__path__", [])) if pkg is not None else []
        if pkg is None:
            pkg = types.ModuleType(pkg_name)
            sys.modules[pkg_name] = pkg
        new_path = [local_env_root] + [p for p in installed_path if p != local_env_root]
        pkg.__path__ = new_path  # bundle dir first, installed package dirs as fallback

    # load the module as usual
    spec = importlib.util.spec_from_file_location("env_module", env_path)
    env_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env_module)
    return env_module

def log_env_hierarchy(env_class, env_path, env_type="training"):
    """
    Logs the env class, its file, and full inheritance hierarchy with file paths.
    env_class: the child env class
    env_path: file path where the child class was loaded from
    env_type: string label, e.g., "training", "evaluation", "resumed_training"
    """
    def get_bases_recursive(cls):
        """Recursively get all base classes with their file paths."""
        info = []
        for base in cls.__bases__:
            try:
                file = inspect.getfile(base)
            except TypeError:
                file = "built-in or unknown"
            info.append(f"{base.__name__} (from {file})")
            # Recurse unless it's object
            if base is not object:
                info.extend(get_bases_recursive(base))
        return info

    hierarchy_info = get_bases_recursive(env_class)
    hierarchy_str = " -> ".join(hierarchy_info) if hierarchy_info else "No parents"

    Journal.log(
        "launch_train_env.py",
        "",
        f"loading {env_type} env {env_class.__name__} (from {env_path}) "
        f"with hierarchy: {hierarchy_str}",
        LogType.INFO,
        throw_when_excep=True
    )

def dummy_step_loop(env):
    global dummy_step_exit_req
    while True:
        if dummy_step_exit_req: 
            return True
        step_ok=env.step(action=env.safe_action) # not a busy loop because of MPC in the step
        if not step_ok:
            return False

if __name__ == "__main__":  

    signal.signal(signal.SIGINT, handle_sigint)

    # Parse command line arguments for CPU affinity
    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")

    parser.add_argument('--run_name', type=str, default=None, help='Name of training run')
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--timeout_ms', type=int, help='Connection timeout after which the script self-terminates', default=60000)
    parser.add_argument('--drop_dir', type=str, help='Directory root where all run data will be dumped')
    parser.add_argument('--run_meta_dir', type=str, default=None,
        help='Resolved IBRIDO run metadata directory to copy into the training/eval run bundle')
    parser.add_argument('--comment', type=str, help='Any useful comment associated with this run', default="")
    parser.add_argument('--seed', type=int, help='Seed', default=1)
    parser.add_argument('--use_cpu',action='store_true', help='If set, all the training (data included) will be performed on CPU')

    parser.add_argument('--db',action='store_true', help='Whether to enable local data logging for the algorithm (reward metrics, etc.)')
    parser.add_argument('--env_db',action='store_true', help='Whether to enable env db data logging on shared mem (e.g. reward metrics are not available for reading anymore)')
    parser.add_argument('--full_env_db',action='store_true', help='Whether to enable detailed episodic data storage (data over single transitions)')
    parser.add_argument('--rmdb',action='store_true', help='Whether to enable remote debug (e.g. data logging on remote servers)')

    parser.add_argument('--tot_tsteps', type=int, help='Total number of timesteps to be collected', default=int(30e6))
    parser.add_argument('--action_repeat', type=int, help='Frame skipping (1-> no skip)', default=1)
    parser.add_argument('--discount_factor', type=float, help='', default=0.99)
    # SAC algorithm hyperparameters (exposed through the common/algorithms/sac_*.yaml configs)
    parser.add_argument('--collection_freq', type=int, help='SAC: env steps collected per iteration', default=1)
    parser.add_argument('--update_freq', type=int, help='SAC: gradient updates per iteration', default=4)
    parser.add_argument('--replay_buffer_n_eps', type=int, help='SAC: replay buffer size in episodes worth of (vectorized) transitions', default=10)
    parser.add_argument('--batch_size', type=int, help='SAC: minibatch size', default=16394)
    parser.add_argument('--lr_policy', type=float, help='SAC: policy learning rate', default=1e-3)
    parser.add_argument('--lr_q', type=float, help='SAC: Q-network learning rate', default=5e-4)
    parser.add_argument('--anneal_entropy', action='store_true', help='SAC: anneal the target entropy from *_start to *_end over training')
    parser.add_argument('--entropy_disc_start', type=float, help='SAC: discrete-action target entropy (start)', default=-0.2)
    parser.add_argument('--entropy_disc_end', type=float, help='SAC: discrete-action target entropy (end)', default=-0.2)
    parser.add_argument('--entropy_cont_start', type=float, help='SAC: continuous-action target entropy (start)', default=-0.5)
    parser.add_argument('--entropy_cont_end', type=float, help='SAC: continuous-action target entropy (end)', default=-0.5)
    # Hybrid SAC hyperparameters (exposed through common/algorithms/hybrid_sac_*.yaml).
    # NOTE the entropy targets here are POSITIVE and expressed as a fraction of the attainable
    # maximum (N*log2 nats) for BOTH branches, unlike the legacy --entropy_* args above, which are
    # negative target log-probabilities. Do not mix the two.
    parser.add_argument('--gumbel_tau', type=float, help='HYBRID SAC: straight-through Gumbel-Sigmoid temperature (backward pass only)', default=0.7)
    parser.add_argument('--disc_grad_mode', type=str, help="HYBRID SAC: 'st_gumbel' or 'exact' (marginalize over the 2^n_binary flag combinations)", default='st_gumbel')
    parser.add_argument('--alpha_cont_init', type=float, help='HYBRID SAC: initial continuous temperature', default=0.2)
    parser.add_argument('--alpha_disc_init', type=float, help='HYBRID SAC: initial discrete temperature', default=1.0)
    parser.add_argument('--lr_alpha', type=float, help='HYBRID SAC: temperature learning rate', default=3e-3)
    parser.add_argument('--target_H_cont_frac', type=float, help='HYBRID SAC: continuous entropy target, as a SIGNED fraction of n_cont*log2. NEGATIVE for a peaked policy; -0.72 matches the legacy -0.5 nats/dim. Max attainable is +0.986', default=-0.72)
    parser.add_argument('--target_H_disc_frac', type=float, help='HYBRID SAC: discrete entropy target, as a fraction of n_binary*log2', default=0.7)
    parser.add_argument('--disc_expl_flip_prob', type=float, help='HYBRID SAC: probability of flipping a binary action on the exploration envs', default=0.25)
    parser.add_argument('--actor_init_std', type=str, help="HYBRID SAC: initial policy std, or 'auto' to derive it from target_H_cont_frac so the continuous branch starts at its entropy target", default='auto')
    parser.add_argument('--use_log_alpha_loss', action='store_true', help='HYBRID SAC: use log_alpha (rather than exp(log_alpha)) as the multiplier in the alpha loss')
    parser.add_argument('--alpha_min', type=str, help="SAC: lower bound on the temperature (anti-windup). 'none' to disable. Legacy default: none", default='none')
    parser.add_argument('--alpha_max', type=str, help="SAC: upper bound on the temperature (anti-windup). Without it, an unreachable entropy target makes alpha diverge exponentially and swamp the Q term. 'none' to disable. Legacy default: none", default='none')
    parser.add_argument('--obs_norm',action='store_true', help='Whether to enable the use of running normalizer in agent')
    parser.add_argument('--obs_rescale',action='store_true', help='Whether to rescale observation depending on their expected range')
    parser.add_argument('--add_weight_norm',action='store_true', help='Whether to add weight normalization to agent interal llayers')
    parser.add_argument('--add_layer_norm',action='store_true', help='Whether to add layer normalization to agent internal llayers')
    parser.add_argument('--add_batch_norm',action='store_true', help='Whether to add batch normalization to agent internal llayers')

    parser.add_argument('--act_rescale_critic',action='store_true', help='Whether to rescale actions provided to critic (if SAC) to be in range [-1, 1]')
    parser.add_argument('--use_period_resets',action='store_true', help='')

    parser.add_argument('--sac',action='store_true', help='Use SAC, otherwise PPO, unless dummy is set')
    parser.add_argument('--hybrid_sac',action='store_true', help='Use hybrid SAC (tanh-Gaussian on the continuous action dims, Bernoulli on the binary ones). Takes precedence over --sac')
    parser.add_argument('--dummy',action='store_true', help='Use dummy agent (useful for testing and debugging environments)')

    parser.add_argument('--dump_checkpoints',action='store_true', help='Whether to dump model checkpoints during training')

    parser.add_argument('--demo_envs_perc', type=float, help='[0, 1]', default=0.0)
    parser.add_argument('--demo_stop_thresh', type=float, default=None, 
        help='Performance hreshold above which demonstration envs should be deactivated.')
    
    parser.add_argument('--expl_envs_perc', type=float, help='[0, 1]', default=0)
    
    parser.add_argument('--use_rnd',action='store_true', help='Whether to use RND for exploration')

    parser.add_argument('--eval',action='store_true', help='Whether to perform an evaluation run')
    parser.add_argument('--n_eval_timesteps', type=int, help='Total number of timesteps to be evaluated', default=int(1e6))
    parser.add_argument('--det_eval',action='store_true', help='Whether to perform a deterministic eval (only action mean is used). Only valid if --eval.')
    parser.add_argument('--allow_expl_during_eval',action='store_true', help='Whether to allow expl envs during evaluation (useful to tune exploration)')
    
    parser.add_argument('--resume',action='store_true', help='Resume a previous training using a checkpoint')
    parser.add_argument('--mpath', type=str, help='Model path to be used for policy evaluation', default=None)
    parser.add_argument('--mname', type=str, help='Model name', default=None)
    parser.add_argument('--override_env',action='store_true', help='Whether to override env when running evaluation')
    
    parser.add_argument('--anomaly_detect',action='store_true', help='Whether to enable anomaly detection (useful for debug)')

    parser.add_argument('--compression_ratio', type=float,
        help='If e.g. 0.8, the fist layer will be of dimension [input_features_size x (input_features_size*compression_ratio)]', default=-1.0)
    parser.add_argument('--actor_lwidth', type=int, help='Actor network layer width', default=128)
    parser.add_argument('--critic_lwidth', type=int, help='Critic network layer width', default=256)
    parser.add_argument('--actor_n_hlayers', type=int, help='Actor network size', default=3)
    parser.add_argument('--critic_n_hlayers', type=int, help='Critic network size', default=4)

    parser.add_argument('--env_fname', type=str, default="twist_tracking_env", help='Training env file name (without extension)')
    parser.add_argument('--env_classname', type=str, default="TwistTrackingEnv", help='Training env class name')
    parser.add_argument('--override_agent_actions',action='store_true', help='Whether to override agent actions with custom ones from shared mem (useful for db)')
    parser.add_argument('--override_agent_refs',action='store_true', help='Whether to override automatically generated agent refs (useful for debug)')
    
    parser.add_argument('--step_while_setup',action='store_true', help='Continue stepping env with default actions while setting up agent, etc..')
    parser.add_argument('--reset_on_init',action='store_true', help='Whether to reset the environment on initialization')

    parser.add_argument('--custom_args_names', nargs='+', default=None,
                            help='list of custom arguments names')
    parser.add_argument('--custom_args_vals', nargs='+', default=None,
                            help='list of custom arguments values')
    parser.add_argument('--custom_args_dtype', nargs='+', default=None,
                            help='list of custom arguments data types')

    args = parser.parse_args()
    args_dict = vars(args)
    custom_opt = generate_custom_arg_dict(args=args)
    args_dict.update(custom_opt)
    if args.run_meta_dir:
        os.environ["IBRIDO_RUN_META_DIR"] = args.run_meta_dir

    if args.eval and args.resume:
        Journal.log("launch_train_env.py",
            "",
            f"Cannot set both --eval and --resume flags. Exiting.",
            LogType.EXCEP,
            throw_when_excep = True)

    deterministic_run(seed=args.seed, torch_det_algos=False)

    anomaly_detect=False
    if args.anomaly_detect:
        torch.autograd.set_detect_anomaly(True)

    if (not args.mpath is None) and (not args.mname is None):
        mpath_full = os.path.join(args.mpath, args.mname)
    else:
        mpath_full=None
    
    env_fname=args.env_fname
    env_classname = args.env_classname
    env_path=""
    env_module=None
    if (not args.eval and not args.resume) or (args.override_env):
        # if starting a fresh traning or overriding env, load from a fresh env from aug_mpc
        env_path = f"aug_mpc_envs.training_envs.{env_fname}"
        env_module = importlib.import_module(env_path)
    else:
        if args.mpath is None:
            Journal.log("launch_train_env.py",
                "",
                f"no mpath provided! Cannot load env. Either provide a mpath or run with --override_env",
                LogType.EXCEP,
                throw_when_excep = True)

        env_path = os.path.join(args.mpath, env_fname + ".py")
        env_module = import_env_module(env_path, local_env_root=args.mpath)
       
    EnvClass = getattr(env_module, env_classname)
    env_type = "training" if not args.eval else "evaluation"
    if args.resume:
        env_type = "resumed_training"
    log_env_hierarchy(EnvClass, env_path, env_type) # db print of env class
    
    env = EnvClass(namespace=args.ns,
            verbose=True,
            vlevel=VLevel.V2,
            use_gpu=not args.use_cpu,
            debug=args.env_db,
            override_agent_refs=args.override_agent_refs,
            timeout_ms=args.timeout_ms,
            env_opts=args_dict)
    if not env.is_ready(): # something went wrong
        exit()
    
    dummy_step_thread = None
    if args.step_while_setup:
        import threading
        # spawn step thread (we don't true parallelization, thread is fine)
        # start the dummy stepping in a separate thread so setup can continue concurrently
        dummy_step_thread = threading.Thread(target=dummy_step_loop, args=(env,), daemon=True)
        dummy_step_thread.start()
    
    # getting some sim info for debugging
    sim_data = {}
    sim_info_shared = SharedEnvInfo(namespace=args.ns,
                is_server=False,
                safe=False)
    sim_info_shared.run()
    sim_info_keys = sim_info_shared.param_keys
    sim_info_data = sim_info_shared.get().flatten()
    for i in range(len(sim_info_keys)):
        sim_data[sim_info_keys[i]] = sim_info_data[i]
    
    # getting come cluster info for debugging
    cluster_data={}
    cluste_info_shared = SharedClusterInfo(namespace=args.ns,
                is_server=False,
                safe=False)
    cluste_info_shared.run()
    cluster_info_keys = cluste_info_shared.param_keys
    cluster_info_data = cluste_info_shared.get().flatten()
    for i in range(len(cluster_info_keys)):
        cluster_data[cluster_info_keys[i]] = cluster_info_data[i]

    custom_args={}
    custom_args["uname_host"]="user_host"
    try:
        username = os.getlogin() # add machine info to db data
        hostname = os.uname().nodename
        user_host = f"{username}@{hostname}"
        custom_args["uname_host"]=user_host
    except:
        pass
    
    algo=None
    if not args.dummy:
        if args.hybrid_sac:
            from aug_mpc.training_algs.sac.hybrid_sac import HybridSAC

            algo = HybridSAC(env=env,
                debug=args.db,
                remote_db=args.rmdb,
                seed=args.seed)
        elif args.sac:
            from aug_mpc.training_algs.sac.sac import SAC

            algo = SAC(env=env,
                debug=args.db,
                remote_db=args.rmdb,
                seed=args.seed)
        else:
            from aug_mpc.training_algs.ppo.ppo import PPO

            algo = PPO(env=env, 
                debug=args.db, 
                remote_db=args.rmdb,
                seed=args.seed)
    else:
        from aug_mpc.training_algs.dummy.dummy import Dummy

        algo=Dummy(env=env, 
                debug=args.db, 
                remote_db=args.rmdb,
                seed=args.seed)

    custom_args.update(args_dict)
    custom_args.update(cluster_data)
    custom_args.update(sim_data)

    run_name=env_classname if args.run_name is None else args.run_name
    algo.setup(run_name=run_name, 
        ns=args.ns,
        verbose=True,
        drop_dir_name=args.drop_dir,
        custom_args=custom_args,
        comment=args.comment,
        eval=args.eval,
        resume=args.resume,
        model_path=mpath_full,
        n_eval_timesteps=args.n_eval_timesteps,
        dump_checkpoints=args.dump_checkpoints,
        norm_obs=args.obs_norm,
        rescale_obs=args.obs_rescale)
    
    full_drop_dir=algo.drop_dir()
    shared_drop_dir = StringTensorServer(length=1, 
        basename="SharedTrainingDropDir", 
        name_space=args.ns,
        verbose=True, 
        vlevel=VLevel.V2, 
        force_reconnection=True)
    shared_drop_dir.run()
    
    while True:
        if not shared_drop_dir.write_vec([full_drop_dir], 0):
            ns=1000000000
            high_resolution_sleep_ns(ns)
            continue
        else:
            break
        
    if args.step_while_setup:
        # stop dummy step thread and give algo authority on step
        dummy_step_exit_req=True
        # wait for thread to join
        if dummy_step_thread is not None:
            dummy_step_thread.join()
        Journal.log("launch_train_env.py",
            "",
            f"Dummy env step thread joined. Moving step authority to algo.",
            LogType.INFO)

    eval=args.eval
    if args.override_agent_actions:
        eval=True
    if not eval:
        while not exit_request:
            if not algo.learn():
                break
    else: # eval phase
        with torch.no_grad(): # no need for grad computation
            while not exit_request:
                if not algo.eval():
                    break
    
    algo.done() # make sure to terminate training properly

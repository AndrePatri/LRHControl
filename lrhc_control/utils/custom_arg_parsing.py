# Ensure custom_args_names, custom_args_vals, and custom_args_dtype have the same length
from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType
import argparse

def generate_custom_arg_dict(args: argparse.Namespace):
    custom_opt = {}
    if args.custom_args_names and args.custom_args_vals and args.custom_args_dtype:
        if not (len(args.custom_args_names) == len(args.custom_args_vals) == len(args.custom_args_dtype)):
            Journal.log("launch_control_cluster.py",
                "",
                f"custom_args_names, custom_args_vals, and custom_args_dtype lengths do not match!",
                LogType.EXCEP,
                throw_when_excep = False)
            exit()
        
        # Build custom_opt dictionary with appropriate data types
        for name, val, dtype in zip(args.custom_args_names, args.custom_args_vals, args.custom_args_dtype):
            try:
                # Convert the value to the appropriate type
                if dtype == "int":
                    custom_opt[name] = int(val)
                elif dtype == "float":
                    custom_opt[name] = float(val)
                elif dtype == "bool":
                    custom_opt[name] = val.lower() in ["true", "1", "yes"]
                else:  # Default is string
                    custom_opt[name] = val
            except ValueError as e:
                Journal.log("custom_arg_parsing.py",
                    "generate_custom_arg_dict",
                    f"Error converting {name} to {dtype}: {e}",
                    LogType.EXCEP,
                    throw_when_excep = False)
                exit()

    return custom_opt



import h5py
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import List, Union, Tuple
import re
import math
import matplotlib.lines as mlines
import argparse
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

# Define a custom colormap (this is an example; adjust as needed)
colors = [
    "#0000FF",  # Blue
    "#00FFFF",  # Cyan
    "#00FF00",  # Green
    "#FFFF00",  # Yellow
    "#FF0000"   # Red
]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

class LRHCPlotter:
    def __init__(self, hdf5_file_path, verbose: bool = True, env_db: bool = False):
        """
        Initialize the LRHCPlotter with the path to the HDF5 file.
        
        Args:
            hdf5_file_path (str): Path to the HDF5 file.
        """
        self._verbose=verbose

        if not os.path.exists(hdf5_file_path):
            raise FileNotFoundError(f"The file '{hdf5_file_path}' does not exist.")
        self.hdf5_file_path = hdf5_file_path
        self.data = {}
        self.attributes = {}  # Dictionary to store file-level attributes
        self.figures = []  # List to store figure objects

        self.map_legend_to_ax = {}  # Will map legend lines to original lines.
        
        if not env_db:
            self._initialize()
    
    def _initialize(self):
        # load env db data
        datasets = self.list_datasets()
        attributes = self.list_attributes()
        self.load_data(dataset_names=datasets, env_idx=args.env_idx)
        self.load_attributes()
        # print("\nDataset names:")
        # print(datasets)
        # print("\n")
        # print("\n Run attributes:")
        # print(attributes)
        # print("\n")

        self.obs_names=list(self.attributes["obs_names"])
        self.action_names=list(self.attributes["action_names"])
        self.sub_trunc_names=list(self.attributes["sub_trunc_names"])
        self.sub_term_names=list(self.attributes["sub_term_names"])
        self.sub_rew_names=list(self.attributes["sub_reward_names"])
        n_envs=self.attributes["n_envs"]
        substepping_dt=self.attributes["substep_dt"]
        action_reps=self.attributes["action_repeat"]
        env_step_dsec=action_reps*substepping_dt
        total_simulated_secs=self.data["n_timesteps_done"]*env_step_dsec
        total_simulated_vec_secs=self.data["n_timesteps_done"]/n_envs
        total_simulated_h=total_simulated_secs/3600.0
        total_simulated_vec_h=total_simulated_vec_secs/3600.0
        total_simulated_d=total_simulated_h/24.0
        total_simulated_vec_d=total_simulated_vec_h/24.0

        self.create_dataset(dataset_name="total_simulated_secs", 
            data=total_simulated_secs)
        self.create_dataset(dataset_name="total_simulated_vec_secs", 
            data=total_simulated_vec_secs)
        self.create_dataset(dataset_name="total_simulated_h", 
            data=total_simulated_h)
        self.create_dataset(dataset_name="total_simulated_vec_h", 
            data=total_simulated_vec_h)
        self.create_dataset(dataset_name="total_simulated_d", 
            data=total_simulated_d)
        self.create_dataset(dataset_name="total_simulated_vec_d", 
            data=total_simulated_vec_d)
        
        # add stats which where not logged explicitly
        self.compute_stats(dataset_name="Actions_avrg",
            stats_dim=1,name="Actions")
        self.compute_stats(dataset_name="AgentTwistRefs_avrg",
            stats_dim=1,name="AgentTwistRefs")
        self.compute_stats(dataset_name="Obs_avrg",
            stats_dim=1,name="Obs")
        self.compute_stats(dataset_name="Obs_avrg",
            stats_dim=1,name="Obs")
        self.compute_stats(dataset_name="Power_avrg",
            stats_dim=1,name="Power")
        self.compute_stats(dataset_name="RhcContactForces_avrg",
            stats_dim=1,name="RhcContactForces")
        self.compute_stats(dataset_name="RhcFailIdx_avrg",
            stats_dim=1,name="RhcFailIdx")
        self.compute_stats(dataset_name="RhcRefsFlag_avrg",
            stats_dim=1,name="RhcRefsIdx")
        self.compute_stats(dataset_name="SubTerminations_avrg",
            stats_dim=1,name="SubTerminations")
        self.compute_stats(dataset_name="SubTruncations_avrg",
            stats_dim=1,name="SubTruncations")
        self.compute_stats(dataset_name="Terminations_avrg",
            stats_dim=1,name="Terminations")
        self.compute_stats(dataset_name="Truncations_avrg",
            stats_dim=1,name="Truncations")
        self.compute_stats(dataset_name="TrackingError_avrg",
            stats_dim=1,name="TrackingError")
        self.compute_stats(dataset_name="TrackingError_avrg",
            stats_dim=1,name="TrackingError")
        
        self.compute_stats(dataset_name="sub_rew_avrg",
                stats_dim=1,name="sub_rew")
        self.compute_stats(dataset_name="tot_rew_avrg",
                stats_dim=1,name="tot_rew")

        self.create_dataset(dataset_name="MechPow_avrg", 
            data=self.data["Power_avrg"][:, :, 1:2])
        self.create_dataset(dataset_name="MechPow_avrg_over_envs", 
            data=self.data["Power_avrg_over_envs"][:, :, 1:2])
        self.create_dataset(dataset_name="MechPow_std_over_envs", 
            data=self.data["Power_std_over_envs"][:, :, 1:2])
        self.create_dataset(dataset_name="MechPow_q1_over_envs", 
            data=self.data["Power_q1_over_envs"][:, :, 1:2])
        self.create_dataset(dataset_name="MechPow_q3_over_envs", 
            data=self.data["Power_q3_over_envs"][:, :, 1:2])
        self.create_dataset(dataset_name="MechPow_median_over_envs", 
            data=self.data["Power_median_over_envs"][:, :, 1:2])
        self.create_dataset(dataset_name="MechPow_max_over_envs", 
            data=self.data["Power_max_over_envs"][:, :, 1:2])
        self.create_dataset(dataset_name="MechPow_min_over_envs", 
            data=self.data["Power_min_over_envs"][:, :, 1:2])
        
        self.create_dataset(dataset_name="CoT_avrg", 
            data=self.data["Power_avrg"][:, :, 0:1])
        self.create_dataset(dataset_name="CoT_avrg_over_envs", 
            data=self.data["Power_avrg_over_envs"][:, :, 0:1])
        self.create_dataset(dataset_name="CoT_std_over_envs", 
            data=self.data["Power_std_over_envs"][:, :, 0:1])
        self.create_dataset(dataset_name="CoT_q1_over_envs", 
            data=self.data["Power_q1_over_envs"][:, :, 0:1])
        self.create_dataset(dataset_name="CoT_q3_over_envs", 
            data=self.data["Power_q3_over_envs"][:, :, 0:1])
        self.create_dataset(dataset_name="CoT_median_over_envs", 
            data=self.data["Power_median_over_envs"][:, :, 0:1])
        self.create_dataset(dataset_name="CoT_max_over_envs", 
            data=self.data["Power_max_over_envs"][:, :, 0:1])
        self.create_dataset(dataset_name="CoT_min_over_envs", 
            data=self.data["Power_min_over_envs"][:, :, 0:1])

        # losses 
        compose_ok=self.compose_datasets(name="qf1_losses",
            datasets_list=["qf1_loss", "qf1_loss_validation"])
        self.compose_datasets(name="qf2_losses",
            datasets_list=["qf2_loss", "qf2_loss_validation"])
        self.compose_datasets(name="actor_losses",
            datasets_list=["actor_loss", "actor_loss_validation"])
        self.compose_datasets(name="alpha_losses",
            datasets_list=["alpha_loss", "alpha_loss_validation"])

        # other training data
        self.compose_datasets(name="qf_vals",
            datasets_list=["qf1_vals_mean", "qf2_vals_mean"])
        self.compose_datasets(name="qf_vals_std",
            datasets_list=["qf1_vals_std", "qf2_vals_std"])
        self.compose_datasets(name="qf_vals_max",
            datasets_list=["qf1_vals_max", "qf2_vals_max"])
        self.compose_datasets(name="qf_vals_min",
            datasets_list=["qf1_vals_min", "qf2_vals_min"])

        # handle sub rewards
        for i in range(len(self.sub_rew_names)):
            sub_rew_name=self.sub_rew_names[i]
            avrg_over_envs_name=sub_rew_name+"_avrg_rew_over_envs"
            std_over_envs_name=sub_rew_name+"_std_rew_over_envs"
            q1_over_envs_name=sub_rew_name+"_q1_rew_over_envs"
            q3_over_envs_name=sub_rew_name+"_q3_rew_over_envs"
            median_over_envs_name=sub_rew_name+"_median_rew_over_envs"
            distr_name=sub_rew_name+"_avrg_rew"
            distr_name_max=sub_rew_name+"_max_rew"
            distr_name_min=sub_rew_name+"_min_rew"

            self.create_dataset(dataset_name=distr_name,
                data=self.data["sub_rew_avrg"][:, :, i:i+1])
            self.create_dataset(dataset_name=distr_name_max,
                data=self.data["sub_rew_max"][:, :, i:i+1])
            self.create_dataset(dataset_name=distr_name_min,
                data=self.data["sub_rew_min"][:, :, i:i+1])
            
            self.create_dataset(dataset_name=avrg_over_envs_name,
                data=self.data["sub_rew_avrg_over_envs"][:, :, i:i+1])
            self.create_dataset(dataset_name=std_over_envs_name,
                data=self.data["sub_rew_std_over_envs"][:, :, i:i+1])
            self.create_dataset(dataset_name=q1_over_envs_name,
                data=self.data["sub_rew_q1_over_envs"][:, :, i:i+1])
            self.create_dataset(dataset_name=q3_over_envs_name,
                data=self.data["sub_rew_q3_over_envs"][:, :, i:i+1])
            self.create_dataset(dataset_name=median_over_envs_name,
                data=self.data["sub_rew_median_over_envs"][:, :, i:i+1])

        if "use_rnd" in attributes:
            if attributes["use_rnd"]:
                self.compose_datasets(name="expl_bonus_proc",
                    datasets_list=["expl_bonus_proc_avrg", "expl_bonus_proc_std"])
                self.compose_datasets(name="expl_bonus_raw",
                    datasets_list=["expl_bonus_raw_avrg", "expl_bonus_raw_std"])

    def list_datasets(self):
        """
        Retrieve and return all dataset names available in the HDF5 file.
        
        Returns:
            list: A list of dataset names in the HDF5 file.
        """
        dataset_names = []
        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                file.visititems(lambda name, obj: dataset_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            return dataset_names
        except Exception as e:
            print(f"Error listing datasets: {e}")
            return dataset_names  # Return an empty list if an error occurs
    
    def list_attributes(self):
        """
        Retrieve and return all attributes from the HDF5 file.
        
        Returns:
            dict: A dictionary of attribute names and their corresponding values.
        """
        attributes = {}
        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                for key, value in file.attrs.items():
                    attributes[key] = value
            return attributes
        except Exception as e:
            print(f"Error listing attributes: {e}")
            return attributes  # Return an empty dictionary if an error occurs

    def load_attributes(self):
        """
        Load all file-level attributes from the HDF5 file and store them in the attributes dictionary.
        """
        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                for key, value in file.attrs.items():
                    self.attributes[key] = value
            # print(f"Loaded attributes: {self.attributes}")
        except Exception as e:
            print(f"Error loading attributes: {e}")

    def load_data(self, dataset_names, env_idx: int = None):
        """
        Load one or more datasets from the HDF5 file.
        
        Args:
            dataset_names (str or list): Name(s) of the datasets to load.
        """
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]  # Convert single string to list

        try:
            with h5py.File(self.hdf5_file_path, 'r') as file:
                for dataset_name in dataset_names:
                    if dataset_name in file:
                        dataset = file[dataset_name]
                        # Check if the dataset is scalar
                        if dataset.shape == ():  # Scalar datasets have an empty shape
                            self.data[dataset_name] = dataset[()]  # Use scalar access
                        else:
                            self.data[dataset_name] = dataset[:]  # Use slicing for arrays
                            if env_idx is not None: # load data just for one specific env
                                if self._verbose:
                                    print(f"Dataset '{dataset_name}' will be loaded only for env {env_idx}. Original shape {self.data[dataset_name].shape[0]}, {self.data[dataset_name].shape[1]}, {self.data[dataset_name].shape[2]}")
                                self.data[dataset_name] = self.data[dataset_name][:, env_idx:env_idx+1, :]
                        if self._verbose:
                            print(f"Dataset '{dataset_name}' with shape {self.data[dataset_name].shape} loaded successfully.")
                    else:
                        if self._verbose:
                            print(f"Warning: Dataset '{dataset_name}' not found in the file.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def add_datas(self, dataset_name, datas: Tuple[np.ndarray],
        avrg: bool = True):
        
        dnames=list(self.data.keys())
        if not dataset_name in dnames:
            print(f"{dataset_name} not found in available data.")
            return False

        if avrg:
            try:
                self.data[dataset_name]=self.data[dataset_name]/len(datas)
            except:
                return False
            
        for i in range(len(datas)):

            if not self.data[dataset_name].shape == (): # not scalar
                n_samples_base=self.data[dataset_name].shape[0]
                n_samples_incoming=datas[i].shape[0]

                # to merge data we need the same number of samples (runs may be of different length)
                n_timesteps=n_samples_base if n_samples_base<=n_samples_incoming else n_samples_incoming

                if self.data[dataset_name].ndim==3:
                    base_data=self.data[dataset_name][0:n_timesteps, :, :]
                    to_be_added=datas[i][0:n_timesteps, :, :]
                elif self.data[dataset_name].ndim==2:
                    base_data=self.data[dataset_name][0:n_timesteps, :]
                    to_be_added=datas[i][0:n_timesteps, :]
                elif self.data[dataset_name].ndim==1:
                    base_data=self.data[dataset_name][0:n_timesteps]
                    to_be_added=datas[i][0:n_timesteps]
                else:
                    raise Exception("add_datas was call on a dataset of dim which is neither 1, 2, or 3")
                
                if not avrg:
                    self.data[dataset_name]=np.concatenate((base_data,to_be_added), axis=1) # add augmented data
                else: # compute average
                    self.data[dataset_name]=base_data+to_be_added/len(datas)
            else:
                print(f"Cannot merge scalar dataset'{dataset_name}'.")

        return True
    
    def create_dataset(self, dataset_name, data: np.ndarray):
        print(f"Created dataset '{dataset_name}' with shape {data.shape}.")
        self.data[dataset_name]=data

    def plot_data(self, dataset_name,
            title="Plot",
            xaxis_dataset_name="", 
            xlabel="Time", 
            ylabel="Intensity", 
            cmap="plasma", # viridis, 
            use_markers=False,
            marker_size: int = 3,
            data_labels = None,
            data_alphas = None,
            data_idxs: List[int] = None,
            distr_std = None, 
            distr_max = None, 
            distr_min = None,
            distr_q1 = None,
            distr_q3 = None,
            distr_p5 = None,
            distr_p95 = None,
            distr_median = None,
            grid_plot: bool = False,
            grid_size: List[int] = None,
            grid_shares_x: bool = True,
            grid_shares_y: bool = True,
            clickable: bool = False):
        """
        Plot the data based on the number of environments in the dataset.
        
        For a single environment, plot a time series. For multiple environments,
        generate a heatmap histogram where the x-axis is time, y-axis is intensity,
        and the color gradient shows frequency. Handles non-finite values by discarding them.
        
        Args:
            dataset_name (str): Name of the dataset to plot.
            xaxis_dataset_name (str): Name of the dataset to use for the x-axis (optional).
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            cmap (str): Colormap for the heatmap (default: "Blues").
            show (bool): Whether to display the plot immediately. Default is True.
        """
        if dataset_name not in self.data:
            print(f"Dataset '{dataset_name}' not loaded. Use 'load_data' first.")
            return

        data_distr_std=None
        data_distr_min=None
        data_distr_max=None
        data_distr_q1=None
        data_distr_q3=None
        data_distr_p5=None
        data_distr_p95=None
        data_distr_median=None
        if distr_std is not None:
            if isinstance(distr_std, str):
                data_distr_std=self.data[distr_std]
        if distr_min is not None:
            if isinstance(distr_min, str):
                data_distr_min=self.data[distr_min]
        if distr_max is not None:
            if isinstance(distr_max, str):
                data_distr_max=self.data[distr_max]
        if distr_q1 is not None:
            if isinstance(distr_q1, str):
                data_distr_q1=self.data[distr_q1]
        if distr_q3 is not None:
            if isinstance(distr_q3, str):
                data_distr_q3=self.data[distr_q3]
        if distr_p5 is not None:
            if isinstance(distr_p5, str):
                data_distr_p5=self.data[distr_p5]
        if distr_p95 is not None:
            if isinstance(distr_p95, str):
                data_distr_p95=self.data[distr_p95]

        if distr_median is not None:
            if isinstance(distr_median, str):
                data_distr_median=self.data[distr_median]

        dataset = self.data[dataset_name]
        n_samples = 1
        n_envs = 1
        n_data = 1

        if dataset.ndim == 3:
            n_samples, n_envs, n_data = dataset.shape
        elif dataset.ndim == 2: 
            n_samples, n_data = dataset.shape
            n_envs=1
            dataset=dataset.reshape(-1, 1, n_data)
        else:
            print(f"Dataset '{dataset_name}' does not have the expected shape (n_samples x n_envs x n_data).")
            return

        fig, axes = None, None  # Initialize figure and axes objects

        data_indexes=list(range(0, n_data)) if data_idxs is None else data_idxs
        labels=[]
        plt_lines=[]
        plt_aux_lines=[]
        if isinstance(ylabel, str):
            ylabels=[ylabel]*len(data_indexes)
        else:
            ylabels=ylabel
        if isinstance(title, str):
            titles=[title]*len(data_indexes)
        else:
            titles=title

        # x axis
        x_datasets=[]
        if isinstance(xaxis_dataset_name, str):
            x_datasets=[xaxis_dataset_name]*len(data_indexes)
        else:
            x_datasets=xaxis_dataset_name

        if xlabel is None:
            xlabel=x_datasets
        if isinstance(xlabel, str):
            xlabels=[xlabel]*len(data_indexes)
        else:
            xlabels=xlabel

        if n_envs == 1:
            
            data = dataset[:, 0, :]  # Extract single environment data
            if data_distr_std is not None and data_distr_std.ndim==3:
                data_distr_std=data_distr_std[:, 0, :]
            if data_distr_max is not None and data_distr_max.ndim==3:
                data_distr_max=data_distr_max[:, 0, :]
            if data_distr_min is not None and data_distr_min.ndim==3:
                data_distr_min=data_distr_min[:, 0, :]
            if data_distr_q1 is not None and data_distr_q1.ndim==3:
                data_distr_q1=data_distr_q1[:, 0, :]
            if data_distr_q3 is not None and data_distr_q3.ndim==3:
                data_distr_q3=data_distr_q3[:, 0, :]
            if data_distr_p5 is not None and data_distr_p5.ndim==3:
                data_distr_p5=data_distr_p5[:, 0, :]
            if data_distr_p95 is not None and data_distr_p95.ndim==3:
                data_distr_p95=data_distr_p95[:, 0, :]
            if data_distr_median is not None and data_distr_median.ndim==3:
                data_distr_median=data_distr_median[:, 0, :]

            if not grid_plot:
                # Time series plot for single environment
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # force scientific notation
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # Force scientific notation

                for i in range(len(data_indexes)):
                    
                    idx=data_indexes[i]
                    label=f"Data {idx+1}" if data_labels is None else data_labels[i]
                    alpha=data_alphas[i] if data_alphas is not None else 1.0
                    labels.append(label)
                    plt_line=None

                    x_dataset=x_datasets[i]
                    if x_dataset not in self.data:
                        print(f"X-axis dataset '{x_dataset}' for data {label} not loaded. Use 'load_data' first.")
                        return
                    xaxis_data = self.data[x_dataset]
                    if xaxis_data.shape != (n_samples, 1):
                        print(f"X-axis dataset '{x_dataset}' for data {label} must have shape ({n_samples}, 1).")
                        return
                    xaxis = xaxis_data[:, 0]  # Extract as 1D array
                    
                    valid_mask = np.logical_and(np.isfinite(data[:, idx]), xaxis[:]>=0)
                    if use_markers:
                        plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], 'o', label=label, markersize=marker_size, alpha=alpha)
                    else:
                        plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], label=label, alpha=alpha)
                    
                    plt_lines.append(plt_line)

                    median_line=None
                    if data_distr_median is not None:
                        color=plt_line.get_color()
                        median_line=ax.plot(xaxis[valid_mask], data_distr_median[valid_mask, idx], '--', 
                            color=color, label=label, markersize=marker_size, alpha=alpha)
                        plt_aux_lines.append(median_line)

                    if data_distr_q1 is not None and data_distr_q3 is not None: # add 25% and 75% quartiles
                        alpha=0.3
                        ax.fill_between(xaxis[valid_mask], 
                                data_distr_q1[valid_mask, idx], data_distr_q3[valid_mask, idx],
                                color=plt_line.get_color(), alpha=alpha, 
                                label="q1/q3")

                    if data_distr_std is not None: # add data distribution std area
                        alpha=0.2
                        ax.fill_between(xaxis[valid_mask], 
                                data[valid_mask, idx] - data_distr_std[valid_mask, idx], data[valid_mask, idx] + data_distr_std[valid_mask, idx],
                                color=plt_line.get_color(), alpha=alpha, 
                                label="std")
                    
                    if data_distr_p5 is not None and data_distr_p95 is not None: # add 25% and 75% quartiles
                        alpha=0.1
                        ax.fill_between(xaxis[valid_mask], 
                                data_distr_p5[valid_mask, idx], data_distr_p95[valid_mask, idx],
                                color=plt_line.get_color(), alpha=alpha, 
                                label="p5/p95")
                    
                    if data_distr_min is not None and data_distr_max is not None: # add min max bounds
                        alpha=0.07
                        ax.fill_between(xaxis[valid_mask], 
                                data_distr_min[valid_mask, idx], data_distr_max[valid_mask, idx],
                                color=plt_line.get_color(), alpha=alpha, 
                                label="min/max")
                
                    if median_line is not None:
                        plt_aux_lines.append(median_line)

                ax.set_title(f"{titles[0]}")
                ax.set_xlabel(xlabels[0])
                ax.set_ylabel(ylabels[0])
                # Create custom legend with lines instead of dots
                legend_lines = [mlines.Line2D([0], [0], color=plt_lines[i].get_color(), lw=4) for i in range(len(data_indexes))]
                legend = ax.legend(legend_lines, labels, ncol=2, handlelength=2)
                # Set pickable property
                for line in legend_lines:
                    line.set_picker(True)
                # legend = ax.legend(ncol=2, markerscale=2)

                legend.set_draggable(True)  # Make the legend draggable
                
                ax.grid(True)

                if clickable:
                    # make legends pickable
                    pickradius=5
                    for legend_line, ax_line in zip(legend.get_lines(), plt_lines):
                        legend_line.set_picker(pickradius)  # Enable picking on the legend line.
                        self.map_legend_to_ax[legend_line] = ax_line
                    
                    def on_pick(event):
                        # On the pick event, find the original line corresponding to the legend
                        # proxy line, and toggle its visibility.
                        legend_line = event.artist

                        # Do nothing if the source of the event is not a legend line.
                        if legend_line not in self.map_legend_to_ax:
                            return

                        ax_line = self.map_legend_to_ax[legend_line]
                        visible = not ax_line.get_visible()
                        ax_line.set_visible(visible)
                        # Change the alpha on the line in the legend, so we can see what lines
                        # have been toggled.
                        legend_line.set_alpha(1.0 if visible else 0.2)
                        fig.canvas.draw()
                    fig.canvas.mpl_connect('pick_event', on_pick)
            else:
        
                if grid_size is None:
                    rows = int(np.ceil(np.sqrt(len(data_indexes))))
                    cols = int(np.ceil(len(data_indexes) / rows))
                else:
                    rows, cols = grid_size

                # Time series plot for single environment (grid)
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=grid_shares_x, sharey=grid_shares_y)
                
                i=0
                for row in range(rows):
                    for col in range(cols):
                        if cols == 1 and not rows==1:
                            ax=axes[row]
                        if rows == 1 and not cols==1:
                            ax=axes[col]
                        if not (rows==1 or cols ==1):
                            ax=axes[row, col]
                        
                        # force scientific notation
                        ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # Force scientific notation

                        idx=data_indexes[i]
                        label=f"Data {idx+1}" if data_labels is None else data_labels[i]
                        alpha=data_alphas[i] if data_alphas is not None else 1.0
                        labels.append(label)
                        plt_line=None

                        x_dataset=x_datasets[i]
                        if x_dataset not in self.data:
                            print(f"X-axis dataset '{x_dataset}' for data {label} not loaded. Use 'load_data' first.")
                            return
                        xaxis_data = self.data[x_dataset]
                        if xaxis_data.shape != (n_samples, 1):
                            print(f"X-axis dataset '{x_dataset}' for data {label} must have shape ({n_samples}, 1).")
                            return
                        xaxis = xaxis_data[:, 0]  # Extract as 1D array


                        valid_mask = np.logical_and(np.isfinite(data[:, idx]), xaxis[:]>=0)
                        if use_markers:
                            plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], 'o', label=label, markersize=marker_size, alpha=alpha)
                        else:
                            plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask, idx], label=label, alpha=alpha)
                        
                        plt_lines.append(plt_line)

                        median_line=None
                        if data_distr_median is not None:
                            color=plt_line.get_color()
                            median_line=ax.plot(xaxis[valid_mask], data_distr_median[valid_mask, idx], '--', 
                                color=color, label=label, markersize=marker_size, alpha=alpha)

                        if data_distr_q1 is not None and data_distr_q3 is not None: # add 25% and 75% quartiles
                            alpha=0.5
                            ax.fill_between(xaxis[valid_mask], 
                                    data_distr_q1[valid_mask, idx], data_distr_q3[valid_mask, idx],
                                    color=plt_line.get_color(), alpha=alpha, 
                                    label="q1/q3")
                            
                        if data_distr_std is not None: # add data distribution std area
                            alpha=0.3
                            ax.fill_between(xaxis[valid_mask], 
                                    data[valid_mask, idx] - data_distr_std[valid_mask, idx], data[valid_mask, idx] + data_distr_std[valid_mask, idx],
                                    color=plt_line.get_color(), alpha=alpha, 
                                    label="std")
                                    
                        if data_distr_p5 is not None and data_distr_p95 is not None: # add 25% and 75% quartiles
                            alpha=0.1
                            ax.fill_between(xaxis[valid_mask], 
                                    data_distr_p5[valid_mask, idx], data_distr_p95[valid_mask, idx],
                                    color=plt_line.get_color(), alpha=alpha, 
                                    label="p5/p95")
                        
                        if data_distr_min is not None and data_distr_max is not None: # add min max bounds
                            alpha=0.07
                            ax.fill_between(xaxis[valid_mask], 
                                    data_distr_min[valid_mask, idx], data_distr_max[valid_mask, idx],
                                    color=plt_line.get_color(), alpha=alpha, 
                                    label="min/max")
                
                        if median_line is not None:
                            plt_aux_lines.append(median_line)

                        ax.set_title(f"{label}")
                        ax.set_xlabel(xlabels[i])
                        ax.set_ylabel(ylabels[i])
                        # Create custom legend with lines instead of dots
                        legend_lines = [mlines.Line2D([0], [0], color=plt_lines[i].get_color(), lw=4)]
                        legend = ax.legend(legend_lines, [label], ncol=2, handlelength=2)
                        # Set pickable property
                        for line in legend_lines:
                            line.set_picker(True)
                        # legend = ax.legend(ncol=2, markerscale=2)

                        legend.set_draggable(True)  # Make the legend draggable
                        
                        ax.grid(True)

                        i+=1

                plt.suptitle(title, y=0.99)
        else:
            # Heatmap histogram for multiple environments
            fig, axes = plt.subplots(1, n_data, figsize=(6*n_data, 4), sharex=grid_shares_x, sharey=grid_shares_y)
            if n_data == 1:
                axes = [axes]  # Ensure axes is iterable for n_data = 1
            
            for i in range(n_data):
                x_dataset=x_datasets[i]
                if x_dataset not in self.data:
                    print(f"X-axis dataset '{x_dataset}' for data {label} not loaded. Use 'load_data' first.")
                    return
                xaxis_data = self.data[x_dataset]
                if xaxis_data.shape != (n_samples, 1):
                    print(f"X-axis dataset '{x_dataset}' for data {label} must have shape ({n_samples}, 1).")
                    return
                xaxis = xaxis_data[:, 0]  # Extract as 1D array
                
                data = dataset[:, :, i]  # Extract data for all environments
                label=f"Data {i+1}" if data_labels is None else data_labels[i]

                # Flatten and filter NaNs from data and corresponding x values
                valid_mask = xaxis > 0  # Valid (finite) mask
                valid_mask = np.logical_and(np.isfinite(data[:, 0]), xaxis[:]>=0)
                valid_time = xaxis[valid_mask]
                valid_values = data[valid_mask, :]

                # nan_count = nan_indexes.size
                # Skip plotting if no valid data exists
                if valid_values.size == 0:
                    print(f"Data {i+1} contains no valid finite values. Skipping.")
                    continue
                
                tiled_values = np.zeros((valid_values.shape[0] * valid_values.shape[1],))
                for j in range(valid_values.shape[1]):
                    tiled_values[j * valid_values.shape[0]:(j * valid_values.shape[0] + valid_values.shape[0])] = valid_values[:, j]
                
                # Compute 2D histogram
                hist_2d, y_edges, x_edges = np.histogram2d(
                    tiled_values.flatten(),
                    np.tile(valid_time, valid_values.shape[1]),  # Filtered time axis
                    bins=(50, 100),  # Time bins and intensity bins
                    density=False,
                )
                
                # Plot the heatmap with white background and blue colormap
                im = axes[i].pcolormesh(
                    x_edges, 
                    y_edges, 
                    hist_2d, 
                    norm=LogNorm(), # linear , symlog
                    cmap=cmap,
                    alpha=0.5,
                    shading="auto", # "nearest", "flat"
                    edgecolors="none",
                    # vmin=500,  # Set the minimum value for the colormap
                    # vmax=2000  # Set the maximum value for the colormap
                # Smooth shading for better visualization
                )
                axes[i].set_title(f"{label}")
                axes[i].set_xlabel(xlabels[i])
                axes[i].set_ylabel(ylabels[i])
                axes[i].grid()
                
                # Add a colorbar to the plot
                fig.colorbar(im, ax=axes[i], label="Frequency")

            
            plt.suptitle(title, y=0.995)
            # fig.subplots_adjust(top=0.6, bottom=0.2)
            # plt.tight_layout()

        # Store the figure in the list
        if fig is not None:
            self.figures.append(fig)
        
    def compute_stats(self, dataset_name, stats_dim = 1, name = None):
        
        data=self.data[dataset_name]
        
        if name is None:
            name=dataset_name
        self.data[name+"_median_over_envs"]=np.median(self.data[dataset_name], 
                                                axis=stats_dim,
                                                keepdims=True)
        self.data[name+"_q1_over_envs"]=np.percentile(self.data[dataset_name],
                                                25, 
                                                axis=stats_dim,
                                                keepdims=True)
        self.data[name+"_q3_over_envs"]=np.percentile(self.data[dataset_name],
                                                75, 
                                                axis=stats_dim,
                                                keepdims=True)
        
        self.data[name+"_p5_over_envs"]=np.percentile(self.data[dataset_name],
                                                5, 
                                                axis=stats_dim,
                                                keepdims=True)
        self.data[name+"_p95_over_envs"]=np.percentile(self.data[dataset_name],
                                                95, 
                                                axis=stats_dim,
                                                keepdims=True)

    def show(self):
        """
        Display all stored plots.
        """
        
        for fig in self.figures:
            fig.show()

        plt.show()

    def get_idx_matching(self, pattern_list, original_list):
        """
        Find the indices of strings in the original list that match any of the patterns.

        Args:
            pattern_list (list of str): List of regex patterns to match. Patterns can include '*' as a wildcard.
            original_list (list of str): List of strings to search in.

        Returns:
            list of int: List of indices of strings in original_list that match any pattern.
            list of str: List of strings in original_list that match any pattern.
        """
        def compile_pattern(pattern):
            """
            Convert a pattern with '*' into a valid regex pattern.
            For example, 'some*pattern' becomes 'some.*pattern'.
            """
            # Escape regex special characters except for '*'
            escaped_pattern = re.escape(pattern).replace(r'\*', '.*')
            return f"^{escaped_pattern}$"

        matching_indices = []
        matching_names = []

        # Precompile patterns into regex objects
        compiled_patterns = [re.compile(compile_pattern(pattern)) for pattern in pattern_list]

        for idx, string in enumerate(original_list):
            for pattern in compiled_patterns:
                if pattern.search(string):  # Check if the compiled pattern matches the string
                    matching_indices.append(idx)
                    break  # Stop checking further patterns for this string

        for i in range(len(matching_indices)):
            matching_names.append(original_list[matching_indices[i]])

        return matching_indices, matching_names
    
    def compose_datasets(self, datasets_list: List[str], name: str):
        """
        Compose multiple datasets along the data dimension (third dimension) and store the result.

        Args:
            datasets_list (List[str]): A list of dataset names to compose.
            name (str): Name for the new composed dataset.

        Raises:
            ValueError: If datasets have incompatible shapes or are not found.
        """
        # Check that all datasets are loaded
        for dataset_name in datasets_list:
            if dataset_name not in self.data:
                print(f"Dataset '{dataset_name}' not loaded. Use 'load_data' first.")
                return False

        # Retrieve the datasets
        datasets = [self.data[dataset_name] for dataset_name in datasets_list]

        # Check compatibility (same number of samples and environments)
        base_shape = datasets[0].shape
        n_samples, n_envs = base_shape[0], base_shape[1] if len(base_shape) > 1 else 1
        for dataset, dataset_name in zip(datasets, datasets_list):
            if len(dataset.shape) == 2:  # Add singleton environment dimension if needed
                dataset = dataset[:, np.newaxis, :]
            if dataset.shape[:2] != (n_samples, n_envs):
                raise ValueError(
                    f"Dataset '{dataset_name}' has incompatible shape {dataset.shape}. "
                    f"Expected {n_samples} samples and {n_envs} environments."
                )

        # Stack datasets along the data dimension
        composed_data = np.concatenate(datasets, axis=-1)

        # Store the new dataset
        self.data[name] = composed_data
        print(f"Composed dataset '{name}' created with shape {composed_data.shape}.")

        return True
class LRHCMultiRunPlotter():

    def __init__(self, hdf5_file_path,
            ablation_attrname: str = None):

        self._base_path=hdf5_file_path
        self._hdf5_files, self._rnames = self.check_hdf5_files(self._base_path)
        
        self._single_run_plotters=[]
        self._single_run_datasets=[]
        self._single_run_attributes=[]

        self._n_runs=len(self._hdf5_files)

        self._x_axis_name="n_timesteps_done"
        for i in range(self._n_runs):
            dataset=self._hdf5_files[i]
            print("########################")
            print(f"Run {self._rnames[i]} ->")
            self._single_run_plotters.append(LRHCPlotter(hdf5_file_path=dataset, verbose=True))
            self._single_run_datasets.append(self._single_run_plotters[i].list_datasets())
            self._single_run_plotters[i].load_attributes()
            self._single_run_plotters[i].load_data(dataset_names=self._single_run_datasets[i])
            self._single_run_attributes.append(list(self._single_run_plotters[i].list_attributes().keys()))
            print("########################")

        self._check_all_attr_are_there()
        self._check_all_datasets_are_there()

        self._dataset_names=self._single_run_datasets[0]
        self._dataset_attributes=self._single_run_attributes[0]

        self._highlight_attr_val_differences()

        # self._final_plotter=self._single_run_plotters[0] # use first plotter for plotting everything
        # self.data=self._final_plotter.data

        # self._x_axis_size=0
        # self._last_idx=np.where(self._single_run_plotters[0].data["n_timesteps_done"]>0)[0][-1]
        # for i in range(self._n_runs-1):
        #     t_axis=self._single_run_plotters[i+1].data["n_timesteps_done"]
        #     last_valid=np.where(t_axis>0)[0][-1]
        #     self._last_idx=last_valid if last_valid < self._last_idx else self._last_idx
        # for i in range(len(self._dataset_names)-1):
        #     datas=()
        #     for j in range(self._n_runs-1):
        #         dset_name=self._dataset_names[i]
        #         dset=self._single_run_plotters[j+1].data[dset_name]
        #         if not self.data[dset_name].shape == (): # not scalar
        #             if dset.ndim < 2:
        #                 datas+=(self._single_run_plotters[j+1].data[dset_name][0:self._last_idx],)
        #             else:
        #                 datas+=(self._single_run_plotters[j+1].data[dset_name][0:self._last_idx, :],)
        #         else:
        #             datas+=(self._single_run_plotters[j+1].data[dset_name],)

        #     ok=self._final_plotter.add_datas(dataset_name=dset_name, datas=datas, avrg=True) # will try to merge data across runs
        #     # computing the average across runs
        #     if not ok:
        #         print(f"Data merge for dataset {dset_name} failed!")
        #     else:
        #         print(f"Dataset '{dset_name}' with shape {self.data[dset_name].shape} loaded successfully.")
        
        self.figures=[]
        self.obs_names=[]
        self.action_names=[]
        self.sub_trunc_names=[]
        self.sub_term_names=[]
        self.sub_rew_names=[]

        # check if obs actions, truncation etc, ... match
        for i in range(self._n_runs):
            self.obs_names.append(self._single_run_plotters[i].obs_names)
            self.action_names.append(self._single_run_plotters[i].action_names)
            self.sub_trunc_names.append(self._single_run_plotters[i].sub_trunc_names)
            self.sub_term_names.append(self._single_run_plotters[i].sub_term_names)
            self.sub_rew_names.append(self._single_run_plotters[i].sub_rew_names)

        self._all_lists_equal(self.obs_names)
        self._all_lists_equal(self.action_names)
        self._all_lists_equal(self.sub_trunc_names)
        self._all_lists_equal(self.sub_term_names)
        self._all_lists_equal(self.sub_rew_names)
        
        # since they matchh, we just use the ones from one dataset
        self.obs_names=self.obs_names[0]
        self.action_names=self.action_names[0]
        self.sub_trunc_names=self.sub_rew_names[0]
        self.sub_term_names=self.sub_term_names[0]
        self.sub_rew_names=self.sub_rew_names[0]

        self._ablation_attrs=[]
        for i in range(self._n_runs):
                self._ablation_attrs.append(f"ablation {i}")
        if ablation_attrname is not None and (ablation_attrname in self._different_attrs_across_runs):
            for i in range(self._n_runs):
                self._ablation_attrs.append(ablation_attrname+ \
                            f": {self._different_attrs_across_runs[ablation_attrname][i]}")
                        

    def _highlight_attr_val_differences(self):
        
        self._attr_values_across_runs={}
        self._attr_values_are_different_across_runs={}
        self._different_attrs_across_runs={}
        attrnames='\n'.join(self._dataset_attributes)
        # print(f"Attribute list: \n {attrnames}\n")

        self.attributes={} # only attributes which are equal

        print(f"################################\n\
        The following different attributes were found:\n")

        for i in range(len(self._dataset_attributes)): # for each attr
            attr_name=self._dataset_attributes[i]
            self._attr_values_across_runs[attr_name]=[]
            self._attr_values_are_different_across_runs[attr_name]=False
            attr_value=self._single_run_plotters[0].attributes[attr_name] # init with
            if isinstance(attr_value, np.ndarray):
                attr_value=attr_value.tolist()

            self._attr_values_across_runs[attr_name].append(attr_value)
            # first run
            for j in range(len(self._single_run_plotters)-1): # for each run
                value=self._single_run_plotters[j+1].attributes[attr_name]
                if isinstance(value, np.ndarray):
                    value=value.tolist()
                if not value==attr_value:
                    self._attr_values_are_different_across_runs[attr_name]=True
                self._attr_values_across_runs[attr_name].append(value)
            if self._attr_values_are_different_across_runs[attr_name]:
                self._different_attrs_across_runs[attr_name]=self._attr_values_across_runs[attr_name]
                print(f"{attr_name}: {self._attr_values_across_runs[attr_name]}\n")
            else:
                self.attributes[attr_name]=self._single_run_plotters[0].attributes[attr_name]

        print(f"################################")
                      
    def check_hdf5_files(self, directory):
        if not os.path.isdir(directory):
            raise ValueError(f"Error: '{directory}' is not a valid directory.")

        # Get a list of all HDF5 files in the directory
        hdf5_files = glob.glob(os.path.join(directory, "*.h5")) + glob.glob(os.path.join(directory, "*.hdf5"))
        file_names = [os.path.splitext(os.path.basename(file))[0] for file in hdf5_files]

        # Check if there are no files or just one file, raise an error
        if len(hdf5_files) < 2:
            raise ValueError("Error: Less than two HDF5 files found in the directory.")
        
        fnames_db_print='\n'.join(file_names)
        print(f"\n[LRHCMultiRunPlotter] Will load runs from datasets: \n {fnames_db_print} \n")

        return hdf5_files, file_names
    
    def _check_all_attr_are_there(self):
        
        self._all_lists_equal(self._single_run_attributes)

    def _check_all_datasets_are_there(self):

        self._all_lists_equal(self._single_run_datasets)

    def _all_lists_equal(self, lst_of_lsts):
        if not lst_of_lsts:  # Handle empty input
            raise ValueError("Error: The input list is empty.")
        
        # Convert each list to a set
        first_set = set(lst_of_lsts[0])  # Take the first list as a reference
        
        for i, lst in enumerate(lst_of_lsts[1:], start=1):  # Compare with the rest
            current_set = set(lst)
            if current_set != first_set:
                missing_in_current = first_set - current_set
                extra_in_current = current_set - first_set
                raise ValueError(
                    f"Error: List at index {i} does not match the reference list.\n"
                    f"Missing elements: {missing_in_current}\n"
                    f"Extra elements: {extra_in_current}"
                )
    
        return True  # All lists are equal

    def list_datasets(self):
        return self._final_plotter.list_datasets()

    def list_attributes(self):
        return self._final_plotter.list_attributes()
    
    def load_data(self, dataset_names, env_idx: int = None):
        pass

    def load_attributes(self):
        pass

    def get_idx_matching(self, pattern_list, original_list):

        return self._final_plotter.get_idx_matching(pattern_list,original_list)
    
    def plot_data(self, dataset_name,
            title="Plot",
            xaxis_dataset_name="", 
            xlabel="Time", 
            ylabel="Intensity", 
            cmap="plasma", # viridis, 
            use_markers=False,
            marker_size: int = 3,
            data_labels = None,
            data_alphas = None,
            data_idxs: List[int] = None,
            distr_std = None, 
            distr_max = None, 
            distr_min = None,
            distr_q1 = None,
            distr_q3 = None,
            distr_p5 = None,
            distr_p95 = None,
            distr_median = None,
            grid_plot: bool = False,
            grid_size: List[int] = None,
            grid_shares_x: bool = True,
            grid_shares_y: bool = True,
            clickable: bool = False):
        
        fig, axes = None, None  # Initialize figure and axes objects

        n_datas=[]
        n_samples=[]
        n_envs=[]
        datasets=[]
        data_distr_std=[]
        data_distr_min=[]
        data_distr_max=[]
        data_distr_q1=[]
        data_distr_q3=[]
        data_distr_p5=[]
        data_distr_p95=[]
        data_distr_median=[]
        
        for j in range(self._n_runs):
            # loop through runs
            data=self._single_run_plotters[j].data
            rname=self._rnames[j]
            if dataset_name not in data:
                print(f"Dataset '{dataset_name}' for run {rname} not loaded. Use 'load_data' first.")
                return

            data_distr_std.append(None)
            data_distr_min.append(None)
            data_distr_max.append(None)
            data_distr_q1.append(None)
            data_distr_q3.append(None)
            data_distr_p5.append(None)
            data_distr_p95.append(None)
            data_distr_median.append(None)
            if distr_std is not None:
                if isinstance(distr_std, str):
                    data_distr_std[j]=self._single_run_plotters[j].data[distr_std]
                    if data_distr_std[j].ndim==3:
                        data_distr_std[j]=self._single_run_plotters[j].data[distr_std][:, 0, :]
            if distr_min is not None:
                if isinstance(distr_min, str):
                    data_distr_min[j]=self._single_run_plotters[j].data[distr_min]
                    if data_distr_min[j].ndim==3:
                        data_distr_min[j]=self._single_run_plotters[j].data[distr_min][:, 0, :]
            if distr_max is not None:
                if isinstance(distr_max, str):
                    data_distr_max[j]=self._single_run_plotters[j].data[distr_max]
                    if data_distr_max[j].ndim==3:
                        data_distr_max[j]=self._single_run_plotters[j].data[distr_max][:, 0, :]
            if distr_q1 is not None:
                if isinstance(distr_q1, str):
                    data_distr_q1[j]=self._single_run_plotters[j].data[distr_q1]
                    if data_distr_q1[j].ndim==3:
                        data_distr_q1[j]=self._single_run_plotters[j].data[distr_q1][:, 0, :]
            if distr_q3 is not None:
                if isinstance(distr_q3, str):
                    data_distr_q3[j]=self._single_run_plotters[j].data[distr_q3]
                    if data_distr_q3[j].ndim==3:
                        data_distr_q3[j]=self._single_run_plotters[j].data[distr_q3][:, 0, :]
            if distr_p5 is not None:
                if isinstance(distr_p5, str):
                    data_distr_p5[j]=self._single_run_plotters[j].data[distr_p5]
                    if data_distr_p5[j].ndim==3:
                        data_distr_p5[j]=self._single_run_plotters[j].data[distr_p5][:, 0, :]
            if distr_p95 is not None:
                if isinstance(distr_p95, str):
                    data_distr_p95[j]=self._single_run_plotters[j].data[distr_p95]
                    if data_distr_p95[j].ndim==3:
                        data_distr_p95[j]=self._single_run_plotters[j].data[distr_p95][:, 0, :]
            if distr_median is not None:
                if isinstance(distr_median, str):
                    data_distr_median[j]=self._single_run_plotters[j].data[distr_median]
                    if data_distr_median[j].ndim==3:
                        data_distr_median[j]=self._single_run_plotters[j].data[distr_median][:, 0, :]

            n_datas.append(1)
            n_samples.append(1)
            n_envs.append(1)
            datasets.append(self._single_run_plotters[j].data[dataset_name])
            if datasets[j].ndim == 3:
                n_samples[j], n_envs[j], n_datas[j] = datasets[j].shape
            elif datasets[j].ndim == 2: 
                n_samples[j], n_datas[j] = datasets[j].shape
                n_envs[j]=1
                datasets[j]=datasets[j].reshape(-1, 1, n_datas[j])
            else:
                print(f"Dataset '{dataset_name}' for run {rname} does not have the expected shape (n_samples x n_envs x n_data).")
                return
            if not n_envs[j]==1:
                print(f"Dataset '{dataset_name}' for run {rname} env dim is not ==1. Multirun distr plots not supported!")
                return
        
        if not len(set(n_datas)) == 1:
            print(f"Not all data in dataset {dataset_name} have the same dimension across runs! Will skip.")
            return
            
        data_indexes=list(range(0, n_datas[0])) if data_idxs is None else data_idxs
        labels=[]
        if isinstance(ylabel, str):
            ylabels=[ylabel]*len(data_indexes)
        else:
            ylabels=ylabel

        # x axis
        x_datasets=[]
        if isinstance(xaxis_dataset_name, str):
            x_datasets=[xaxis_dataset_name]*self._n_runs
        else:
            x_datasets=xaxis_dataset_name

        if xlabel is None:
            xlabel=x_datasets
        if isinstance(xlabel, str):
            xlabels=[xlabel]*len(data_indexes)
        else:
            xlabels=xlabel

        if grid_plot:
            # data is visualized independently for each run
            rows = len(data_indexes)
            cols = self._n_runs
        else:
            # multiple runs are stacked on top of the same data
            rows = 1
            cols = len(data_indexes)
    
        fig, axes = plt.subplots(rows, cols, 
            figsize=(5 * cols, 4 * rows), sharex=grid_shares_x, sharey=grid_shares_y)
        
        if rows*cols==1:
            axes=[axes]

        for i in range(len(data_indexes)):
            # loop over data
            for run in range(self._n_runs):
                data=datasets[run][:, 0, i] 

                if grid_plot:
                    ax=axes[i, run]
                else:
                    ax=axes[i]

                # force scientific notation
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # Force scientific notation
    
                idx=data_indexes[i]
                dlabel=f"Data {idx+1}" if data_labels is None else data_labels[i]
                alpha=data_alphas[i] if data_alphas is not None else 1.0
                plt_line=None

                x_dataset=x_datasets[run]
                if x_dataset not in self._single_run_plotters[run].data:
                    print(f"X-axis dataset '{x_dataset}' for data {dlabel}, run {self._rnames[run]} not loaded. Use 'load_data' first.")
                    return
                xaxis_data = self._single_run_plotters[run].data[x_dataset]
                if xaxis_data.shape != (n_samples[run], 1):
                    print(f"X-axis dataset '{x_dataset}' for data {dlabel}, run {self._rnames[run]} must have shape ({n_samples[run]}, 1) but got {xaxis_data.shape}")
                    return
                xaxis = xaxis_data[:, 0]  # Extract as 1D array

                valid_mask = np.logical_and(np.isfinite(data), xaxis[:]>=0)
                if use_markers:
                    plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask], 'o', label=self._ablation_attrs[run], markersize=marker_size, alpha=alpha)
                else:
                    plt_line, = ax.plot(xaxis[valid_mask], data[valid_mask], label=self._ablation_attrs[run], alpha=alpha)
                
                median_line=None
                if data_distr_median[run] is not None:
                    color=plt_line.get_color()
                    median_line=ax.plot(xaxis[valid_mask], data_distr_median[run][valid_mask, idx], '--', 
                        color=color, label=self._ablation_attrs[run], markersize=marker_size, alpha=alpha)

                if data_distr_q1[run] is not None and data_distr_q3[run] is not None: # add 25% and 75% quartiles
                    alpha=0.5
                    ax.fill_between(xaxis[valid_mask], 
                            data_distr_q1[run][valid_mask, idx], data_distr_q3[run][valid_mask, idx],
                            color=plt_line.get_color(), alpha=alpha, 
                            label="q1/q3")
                
                if data_distr_p5[run] is not None and data_distr_p95[run] is not None: # add 25% and 75% quartiles
                    alpha=0.5
                    ax.fill_between(xaxis[valid_mask], 
                            data_distr_p5[run][valid_mask, idx], data_distr_p95[run][valid_mask, idx],
                            color=plt_line.get_color(), alpha=alpha, 
                            label="p1/p5")

                if data_distr_std[run] is not None: # add data distribution std area
                    alpha=0.3
                    ax.fill_between(xaxis[valid_mask], 
                            data[valid_mask] - data_distr_std[run][valid_mask, idx], data[valid_mask] + data_distr_std[run][valid_mask, idx],
                            color=plt_line.get_color(), alpha=alpha, 
                            label="± 1 std")
                            
                if data_distr_min[run] is not None and data_distr_max[run] is not None: # add min max bounds
                    alpha=0.15
                    ax.fill_between(xaxis[valid_mask], 
                            data_distr_min[run][valid_mask, idx], data_distr_max[run][valid_mask, idx],
                            color=plt_line.get_color(), alpha=alpha, 
                            label="min/max")
                
                if i==0:
                    ax.set_title(f"{self._ablation_attrs[run]}")

                if i==(len(data_indexes)-1):
                    ax.set_xlabel(xlabels[i])
                
                if run==(self._n_runs-1):
                    ax.yaxis.set_label_position("right")
                    # ax.yaxis.set_label_position("right")
                    ax.set_ylabel(ylabels[i])
                
                if run==0:
                    ax.set_ylabel(dlabel+": ", rotation=0, labelpad=20, 
                        fontsize=11)

                # Create custom legend with lines instead of dots
                legend_lines = [mlines.Line2D([0], [0], color=plt_line.get_color(), lw=4)]
                legend = ax.legend(legend_lines, self._ablation_attrs[run], ncol=2, handlelength=2)
                # Set pickable property
                for line in legend_lines:
                    line.set_picker(True)
                # legend = ax.legend(ncol=2, markerscale=2)

                legend.set_draggable(True)  # Make the legend draggable
                
                ax.grid(True)

        plt.suptitle(title, y=0.99)

        # Store the figure in the list
        if fig is not None:
            self.figures.append(fig)
            
    def compose_datasets(self, datasets_list: List[str], name: str):
        self._final_plotter.compose_datasets(datasets_list, name)

    def show(self):
        for fig in self.figures:
            fig.show()

        plt.show()
    
    def create_dataset(self, dataset_name, data: np.ndarray):
        print(f"Created dataset '{dataset_name}' with shape {data.shape}.")
        self._final_plotter.create_dataset(dataset_name, data)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--env_db',action='store_true', help='')
    parser.add_argument('--expl',action='store_true', help='whether to plot db data from expl env (if any)')
    parser.add_argument('--demo',action='store_true', help='whether to plot db data from demo env (if any)')
    parser.add_argument('--env_idx',type=int, help='', default=None)
    parser.add_argument('--data_path',type=str, help='full path to dataset to plot')
    parser.add_argument('--running_obs_stats',action='store_true', help='whether to plot running stats used for obs normalization')
    parser.add_argument('--actions_stats',action='store_true', help='')
    parser.add_argument('--xdset',type=str, help='x dataset to be used for x axis', default="n_timesteps_done")

    parser.add_argument('--multirun',action='store_true', help='plot comparative results (if env db across envs, otherwise across runs)')
    parser.add_argument('--ablation_attr',type=str, help='attribute wrt ablation study was run (if multurun)', default=None)

    args = parser.parse_args()

    path = args.data_path
    xaxis_dataset_name=args.xdset
    # plot some data
    marker_size=1

    if not args.env_db:
        
        if args.multirun:
            plotter = LRHCMultiRunPlotter(hdf5_file_path=path,
                                ablation_attrname=args.ablation_attr)
            
        else:
            # load training data
            plotter = LRHCPlotter(hdf5_file_path=path)
        
        xlabel=xaxis_dataset_name
        
        plotter.plot_data(dataset_name="qf1_losses", title="qf1 loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bellman error",
            data_labels=["qf1 training loss", "qf1 validation loss"],
            data_alphas=[0.9, 0.5],
            use_markers=False,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="qf2_losses", title="qf2 loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bellman error",
            data_labels=["qf2 training loss", "qf2 validation loss"],
            data_alphas=[0.9, 0.5],
            use_markers=False,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="actor_losses", title="actor_loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="[]",
            data_labels=["training", "validation"],
            data_alphas=[0.9, 0.5],
            use_markers=False,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="alpha_losses", title="alpha_loss", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="[]",
            data_labels=["training", "validation"],
            data_alphas=[0.6, 0.2],
            use_markers=False,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="qf1_vals_mean", 
            title="Q values stats over tr. batches", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="Q val.",
            data_labels=["qf"],
            use_markers=False,
            marker_size=marker_size,
            distr_std="qf1_vals_std",
            distr_max=None,
            distr_min=None)
        
        # sub rewards
        plotter.plot_data(dataset_name="sub_rew_avrg_over_envs", 
            title=f"scaled sub returns stats over envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="",
            data_labels=plotter.sub_rew_names,
            use_markers=False,
            marker_size=marker_size,
            distr_std=None,
            distr_max=None,
            distr_min=None,
            distr_q1="sub_rew_q1_over_envs",
            distr_q3="sub_rew_q3_over_envs",
            distr_p5="sub_rew_p5_over_envs",
            distr_p95="sub_rew_p95_over_envs",
            distr_median="sub_rew_median_over_envs") 
        plotter.plot_data(dataset_name="sub_rew_avrg_over_envs", 
            title=f"scaled sub returns stats over envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="",
            data_labels=plotter.sub_rew_names,
            use_markers=False,
            marker_size=marker_size,
            distr_std=None,
            distr_max=None,
            distr_min=None,
            distr_q1="sub_rew_q1_over_envs",
            distr_q3="sub_rew_q3_over_envs",
            distr_p5="sub_rew_p5_over_envs",
            distr_p95="sub_rew_p95_over_envs",
            distr_median="sub_rew_median_over_envs",
            grid_plot=True,
            grid_shares_y=False,
            grid_size=[1, len(plotter.sub_rew_names)]) 
        plotter.plot_data(dataset_name="sub_rew_avrg", 
            title=f"scaled sub returns distribution across envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="",
            data_labels=plotter.sub_rew_names,
            grid_shares_y=False)
        
        # tot reward
        plotter.plot_data(dataset_name="tot_rew_avrg_over_envs", 
            title=f"scaled return stats over envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="",
            data_labels=["tot_rew"],
            use_markers=False,
            marker_size=marker_size,
            distr_std=None,
            distr_max=None,
            distr_min=None,
            distr_q1="tot_rew_q1_over_envs",
            distr_q3="tot_rew_q3_over_envs",
            distr_p5="tot_rew_p5_over_envs",
            distr_p95="tot_rew_p95_over_envs",
            distr_median="tot_rew_median_over_envs") 
        plotter.plot_data(dataset_name="tot_rew_avrg", 
            title=f"scaled return distribution across envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="",
            data_labels=plotter.sub_rew_names,
            grid_shares_y=False)

        # env data 
        plotter.plot_data(dataset_name="env_step_rt_factor", title="env_step_rt_factor", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            use_markers=True,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="ep_tsteps_env_distr", title="ep_tsteps_env_distribution", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            use_markers=True,
            marker_size=marker_size)
        
        plotter.plot_data(dataset_name="SubTruncations_avrg_over_envs", title="SubTruncations_avrg_over_envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bool",
            data_labels=plotter.sub_trunc_names,
            use_markers=True,
            marker_size=marker_size,
            distr_std=None,
            distr_max=None, # tot_rew_max_over_envs
            distr_min=None,
            distr_median="SubTruncations_median_over_envs",
            distr_q1=None,
            distr_q3=None,
            grid_plot=True,
            grid_shares_y=True,
            grid_size=[1, len(plotter.sub_trunc_names)])
        plotter.plot_data(dataset_name="SubTerminations_avrg_over_envs", title="SubTerminations_avrg_over_envs", 
            xaxis_dataset_name=xaxis_dataset_name,
            xlabel=xlabel,
            ylabel="bool",
            data_labels=plotter.sub_term_names,
            use_markers=True,
            marker_size=marker_size,
            distr_std=None,
            distr_max=None, # tot_rew_max_over_envs
            distr_min=None,
            distr_median="SubTerminations_median_over_envs",
            distr_q1=None,
            distr_q3=None,
            grid_plot=True,
            grid_shares_y=True,
            grid_size=[1, len(plotter.sub_term_names)])
        
        # rnd
        if "use_rnd" in plotter.attributes:
            if plotter.attributes["use_rnd"]:
                plotter.plot_data(dataset_name="expl_bonus_proc", title="expl_bonus_proc", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_alphas=[0.3, 0.3],
                    data_labels=["expl_bonus_proc_avrg", "expl_bonus_proc_std"])
                
                plotter.plot_data(dataset_name="expl_bonus_raw", title="expl_bonus_raw", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_alphas=[0.3, 0.3],
                    data_labels=["expl_bonus_raw_avrg", "expl_bonus_raw_std"])
        
        # plotter.plot_data(dataset_name="CoT_avrg_over_envs", 
        #         title=f"CoT", 
        #         xaxis_dataset_name=xaxis_dataset_name,
        #         xlabel=xlabel,
        #         ylabel="[]",
        #         data_labels="CoT",
        #         use_markers=False,
        #         marker_size=marker_size,
        #         distr_std="CoT_std_over_envs",
        #         distr_max=None, # tot_rew_max_over_envs
        #         distr_min=None,
        #         distr_q1="CoT_q1_over_envs",
        #         distr_q3="CoT_q3_over_envs",
        #         distr_median="CoT_median_over_envs")
        # plotter.plot_data(dataset_name="CoT_avrg", 
        #     title=f"CoT distribution", 
        #     xaxis_dataset_name=xaxis_dataset_name,
        #     xlabel=xlabel) # distribution
        
        # plotter.plot_data(dataset_name="MechPow_avrg_over_envs", 
        #         title=f"Mech. power", 
        #         xaxis_dataset_name=xaxis_dataset_name,
        #         xlabel=xlabel,
        #         ylabel="[]",
        #         data_labels="Pow",
        #         use_markers=False,
        #         marker_size=marker_size,
        #         distr_std="MechPow_std_over_envs",
        #         distr_max=None, # tot_rew_max_over_envs
        #         distr_min=None,
        #         distr_q1="MechPow_q1_over_envs",
        #         distr_q3="MechPow_q3_over_envs",
        #         distr_median="MechPow_median_over_envs")
        # plotter.plot_data(dataset_name="MechPow_avrg", 
        #     title=f"Mechanical power distribution", 
        #     xaxis_dataset_name=xaxis_dataset_name,
        #     xlabel=xlabel) # distribution
        
        plotter.plot_data(dataset_name="Power_avrg_over_envs", 
                title=f"Power db data", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=None,
                ylabel=["[]", "[W]"],
                data_labels=["CoT", "Mech.P."],
                use_markers=False,
                marker_size=marker_size,
                distr_std=None,
                distr_max=None, # tot_rew_max_over_envs
                distr_min=None,
                distr_p5="Power_p5_over_envs",
                distr_p95="Power_p95_over_envs",
                distr_q1="Power_q1_over_envs",
                distr_q3="Power_q3_over_envs",
                distr_median="Power_median_over_envs",
                grid_plot=True,
                grid_size=[1, 2],
                grid_shares_y=False)
        
        plotter.plot_data(dataset_name="TrackingError_avrg_over_envs", 
                title=f"Tracking error", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                ylabel=["m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s"],
                data_labels=["lin_x", "lin_y", "lin_z", "omega_x", "omega_y", "omega_z"],
                use_markers=False,
                marker_size=marker_size,
                distr_std=None,
                distr_max=None, # tot_rew_max_over_envs
                distr_min=None,
                distr_p5="TrackingError_p5_over_envs",
                distr_p95="TrackingError_p95_over_envs",
                distr_q1="TrackingError_q1_over_envs",
                distr_q3="TrackingError_q3_over_envs",
                distr_median="TrackingError_median_over_envs",
                grid_plot=True,
                grid_size=[2, 3])
        
        # actions stats
        if args.actions_stats:
            patterns_lin_v=["v*_*cmd"]
            idxs_lin_v,selected_lin_v=plotter.get_idx_matching(patterns_lin_v, plotter.action_names)
            patterns_ang_v=["*omega_cmd*"]
            idxs_ang_v,selected_ang_v=plotter.get_idx_matching(patterns_ang_v, plotter.action_names)
            idxs=idxs_lin_v+idxs_ang_v
            selected=selected_lin_v+selected_ang_v
            ylabels=["[m/s]"]*len(idxs_lin_v)
            ylabels+=["[rad/s]"]*len(idxs_ang_v)
            if len(idxs_lin_v)>0 and len(idxs_ang_v)>0:
                plotter.plot_data(dataset_name="Actions_avrg_over_envs", 
                    title=f"Actions - twist commands", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=None,
                    ylabel=ylabels,
                    data_labels=selected,
                    data_idxs=idxs,
                    use_markers=False,
                    marker_size=marker_size,
                    distr_std=None,
                    distr_max="Actions_max_over_envs",
                    distr_min="Actions_min_over_envs",
                    distr_q1="Actions_q1_over_envs",
                    distr_q3="Actions_q3_over_envs",
                    distr_p5="Actions_p5_over_envs",
                    distr_p95="Actions_p95_over_envs",
                    distr_median="Actions_median_over_envs",
                    grid_plot=True,
                    grid_size=[1, len(idxs)])
            
            patterns=["*contact_flag*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.action_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="Actions_avrg_over_envs", 
                    title=f"Actions - contact flags", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=None,
                    ylabel=["[m]"]*len(idxs),
                    data_labels=selected,
                    data_idxs=idxs,
                    use_markers=False,
                    marker_size=marker_size,
                    distr_std=None,
                    distr_max="Actions_max_over_envs",
                    distr_min="Actions_min_over_envs",
                    distr_q1="Actions_q1_over_envs",
                    distr_q3="Actions_q3_over_envs",
                    distr_p5="Actions_p5_over_envs",
                    distr_p95="Actions_p95_over_envs",
                    distr_median="Actions_median_over_envs",
                    grid_plot=True,
                    grid_size=[1, len(idxs)])
                
            patterns=["*flight_apex*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.action_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="Actions_avrg_over_envs", 
                    title=f"Actions - apex heights", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=None,
                    ylabel=["[m]"]*len(idxs),
                    data_labels=selected,
                    data_idxs=idxs,
                    use_markers=False,
                    marker_size=marker_size,
                    distr_std=None,
                    distr_max="Actions_max_over_envs",
                    distr_min="Actions_min_over_envs",
                    distr_q1="Actions_q1_over_envs",
                    distr_q3="Actions_q3_over_envs",
                    distr_p5="Actions_p5_over_envs",
                    distr_p95="Actions_p95_over_envs",
                    distr_median="Actions_median_over_envs",
                    grid_plot=True,
                    grid_size=[1, len(idxs)])

            patterns=["*flight_len*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.action_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="Actions_avrg_over_envs", 
                    title=f"Actions - flight lengths", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=None,
                    ylabel=["[m]"]*len(idxs),
                    data_labels=selected,
                    data_idxs=idxs,
                    use_markers=False,
                    marker_size=marker_size,
                    distr_std=None,
                    distr_max="Actions_max_over_envs",
                    distr_min="Actions_min_over_envs",
                    distr_q1="Actions_q1_over_envs",
                    distr_q3="Actions_q3_over_envs",
                    distr_p5="Actions_p5_over_envs",
                    distr_p95="Actions_p95_over_envs",
                    distr_median="Actions_median_over_envs",
                    grid_plot=True,
                    grid_size=[1, len(idxs)])
            
            patterns=["*flight_end*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.action_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="Actions_avrg_over_envs", 
                    title=f"Actions - flight end height", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=None,
                    ylabel=["[m]"]*len(idxs),
                    data_labels=selected,
                    data_idxs=idxs,
                    use_markers=False,
                    marker_size=marker_size,
                    distr_std=None,
                    distr_max="Actions_max_over_envs",
                    distr_min="Actions_min_over_envs",
                    distr_q1="Actions_q1_over_envs",
                    distr_q3="Actions_q3_over_envs",
                    distr_p5="Actions_p5_over_envs",
                    distr_p95="Actions_p95_over_envs",
                    distr_median="Actions_median_over_envs",
                    grid_plot=True,
                    grid_size=[1, len(idxs)])
            
            patterns=["*phase_freq*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.action_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="Actions_avrg_over_envs", 
                    title=f"Actions - phase frequency", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=None,
                    ylabel=["[flights/mpc_step]"]*len(idxs),
                    data_labels=selected,
                    data_idxs=idxs,
                    use_markers=False,
                    marker_size=marker_size,
                    distr_std=None,
                    distr_max="Actions_max_over_envs",
                    distr_min="Actions_min_over_envs",
                    distr_q1="Actions_q1_over_envs",
                    distr_q3="Actions_q3_over_envs",
                    distr_p5="Actions_p5_over_envs",
                    distr_p95="Actions_p95_over_envs",
                    distr_median="Actions_median_over_envs",
                    grid_plot=True,
                    grid_size=[1, len(idxs)])
            
            patterns=["*phase_offset*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.action_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="Actions_avrg_over_envs", 
                    title=f"Actions - phase offset", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=None,
                    ylabel=[""]*len(idxs),
                    data_labels=selected,
                    data_idxs=idxs,
                    use_markers=False,
                    marker_size=marker_size,
                    distr_std=None,
                    distr_max="Actions_max_over_envs",
                    distr_min="Actions_min_over_envs",
                    distr_q1="Actions_q1_over_envs",
                    distr_q3="Actions_q3_over_envs",
                    distr_p5="Actions_p5_over_envs",
                    distr_p95="Actions_p95_over_envs",
                    distr_median="Actions_median_over_envs",
                    grid_plot=True,
                    grid_size=[1, len(idxs)])

        if args.running_obs_stats:
            # obs stats
            # gravity vecs
            patterns=["gn_*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - gravity vec", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - gravity vec", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # joint pos
            patterns=["q_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - meas joint q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - meas joint q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # joint vel
            patterns=["v_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - meas joint v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - meas joint v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # cmd efforts
            patterns=["rhc_cmd_q_*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc cmd q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc cmd q", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # cmd efforts
            patterns=["rhc_cmd_v_*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc cmd v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc cmd v", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # cmd efforts
            patterns=["rhc_cmd_eff_*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc cmd effort", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc cmd effort", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # estimated contact forces
            patterns=["fc_contact*"]
            idxs, selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - est. contact f", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - est. contact f", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # mpc fail idx
            patterns=["rhc_fail*"]
            idxs, selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - MPC fail index", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - MPC fail index", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # rhc flight info
            patterns=["flight_*"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - rhc flight info", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - rhc flight info", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # linvel
            patterns=["linvel_*_base_loc"]
            idxs, selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - linvel (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - linvel (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # omega
            patterns=["omega_*_base_loc"]
            idxs, selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - omega (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - omega (meas/ref)", 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs,
                clickable=True)
            
            # clock (if any)
            patterns=["clock*"]
            idxs, selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - clock", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
                plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - clock", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
            
            # actions buffer stats (if used)
            patterns=["*_prev_act"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer - prev cmds", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
                plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer - prev cmds", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
            patterns=["*_avrg_act"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer - mean cmds over window", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
                plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer - mean cmds over window", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
            patterns=["*_std_act"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer ", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
                plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer ", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
            
            # full actions buffer (if used)
            patterns=["*_m*_act"]
            idxs,selected=plotter.get_idx_matching(patterns, plotter.obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name="running_mean_obs", title="running_mean_obs - action buffer ", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
                plotter.plot_data(dataset_name="running_std_obs", title="running_std_obs - action buffer ", 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs,
                    clickable=True)
                    
    else:
        # plot env debug data (over single steps)
        dset_suffix="" if not args.expl else "_expl"
        if args.demo:
            dset_suffix="_demo"

        # load env db data
        plotter = LRHCPlotter(hdf5_file_path=path)
        datasets = plotter.list_datasets()
        attributes = plotter.list_attributes()
        plotter.load_data(dataset_names=datasets, env_idx=args.env_idx)
        plotter.load_attributes()
        print("\nDataset names:")
        print(datasets)
        print("\n")
        print("\n Run attributes:")
        print(attributes)
        print("\n")

        n_eps=int(plotter.attributes["ep_vec_freq"])
        obs_names=list(plotter.attributes["Obs_data_names"])
        actions_names=list(plotter.attributes["Actions_data_names"])
        contact_forces_names=list(plotter.attributes["RhcContactForces_data_names"])
        twist_refs_names=list(plotter.attributes["AgentTwistRefs_data_names"])
        rhc_refs_names=list(plotter.attributes["RhcRefsFlag_data_names"])
        sub_term_names=list(plotter.attributes["SubTerminations_data_names"])
        sub_trunc_names=list(plotter.attributes["SubTruncations_data_names"])
        sub_reward_names=list(plotter.attributes["sub_reward_names"])
        pow_names=list(plotter.attributes["Power_data_names"])
        track_err_names=list(plotter.attributes["TrackingError_data_names"])

        xlabel="env_step"
        xaxis_dataset_name=None
        marker_size=2
        for ep_idx in range(n_eps):
            ep_prefix=f"ep_{ep_idx}_"
            
            obs_datasetname=ep_prefix+"Obs"+dset_suffix
            actions_datasetname=ep_prefix+"Actions"+dset_suffix
            # gravity vec
            patterns=["gn_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - gravity vec"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - gravity vec"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # joint pos
            patterns=["q_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - meas joint q"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
        
            # joint vel
            patterns=["v_jnt_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - meas joint v"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd effort
            patterns=["rhc_cmd_q_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc cmd q"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd effort
            patterns=["rhc_cmd_v_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc cmd v"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # cmd effort
            patterns=["rhc_cmd_eff_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc cmd effort"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # estimated contact forces
            patterns=["fc_contact*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - est. contact f"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # mpc fail idx
            patterns=["rhc_fail*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - MPC fail index"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # rhc flight info
            patterns=["flight_*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - rhc flight info"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # linvel
            patterns=["linvel_*_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - linvel (meas/ref)"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            # omega
            patterns=["omega_*_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            plotter.plot_data(dataset_name=obs_datasetname, 
                title=ep_prefix+"obs - omega (meas/ref)"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)

            # clock (if any)
            patterns=["clock*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=obs_datasetname, 
                    title=ep_prefix+"clock"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)
                
            # actions buffer
            patterns=["*_prev_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=obs_datasetname, 
                    title=ep_prefix+"obs - action buffer - prev cmds"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)
            patterns=["*_avrg_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=obs_datasetname, 
                    title=ep_prefix+"obs - action buffer - mean cmds over window"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)
            patterns=["*_std_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=obs_datasetname, 
                    title=ep_prefix+"obs - action buffer "+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)

            patterns=["*_m*_act"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=obs_datasetname, 
                    title=ep_prefix+"obs - action buffer - full action history buffer"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=True,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)
                
            # actions
            plotter.plot_data(dataset_name=actions_datasetname, 
                title=ep_prefix+"actions - all"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=actions_names,
                data_idxs=None)
            # contact actions
            patterns=["*contact_flag*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=actions_datasetname, 
                    title=ep_prefix+"actions - contact flag actions only"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=False,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)
            
            patterns=["*phase_freq*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=actions_datasetname, 
                    title=ep_prefix+"actions - step frequency only [flights/mpc_steps]"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=False,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)
            
            patterns=["*phase_offset*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=actions_datasetname, 
                    title=ep_prefix+"actions - step offset only [mpc_steps]"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=False,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)

            patterns=["*flight_*"]
            idxs,selected=plotter.get_idx_matching(patterns, actions_names)
            if len(idxs)>0:
                plotter.plot_data(dataset_name=actions_datasetname, 
                    title=ep_prefix+"actions - flight params actions only"+dset_suffix, 
                    xaxis_dataset_name=xaxis_dataset_name,
                    xlabel=xlabel,
                    use_markers=False,
                    marker_size=marker_size,
                    data_labels=selected,
                    data_idxs=idxs)

            # sub terminations
            plotter.plot_data(dataset_name=ep_prefix+"SubTerminations"+dset_suffix, 
                title=ep_prefix+"SubTerminations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=sub_term_names,
                data_idxs=None)
            
            
            # sub terminations
            plotter.plot_data(dataset_name=ep_prefix+"SubTruncations"+dset_suffix, 
                title=ep_prefix+"SubTruncations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=sub_trunc_names,
                data_idxs=None)
            
            # terminations
            plotter.plot_data(dataset_name=ep_prefix+"Terminations"+dset_suffix, 
                title=ep_prefix+"Terminations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=["is_terminal"],
                data_idxs=None)
            
            # truncations
            plotter.plot_data(dataset_name=ep_prefix+"Truncations"+dset_suffix, 
                title=ep_prefix+"Truncations"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=["is_truncated"],
                data_idxs=None)
            
            # sub rewards
            plotter.plot_data(dataset_name=ep_prefix+"sub_rew"+dset_suffix, 
                title=ep_prefix+"sub_rew"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=sub_reward_names,
                data_idxs=None)
            
            # tot rewards
            plotter.plot_data(dataset_name=ep_prefix+"tot_rew"+dset_suffix, 
                title=ep_prefix+"tot reward"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=["tot_reward"],
                data_idxs=None)

            # agent twist refs
            plotter.plot_data(dataset_name=ep_prefix+"AgentTwistRefs"+dset_suffix, 
                title=ep_prefix+"agent refs"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=twist_refs_names,
                data_idxs=None)

            # other custom data

            plotter.plot_data(dataset_name=ep_prefix+"Power"+dset_suffix, 
                title=ep_prefix+"Power"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=pow_names,
                data_idxs=None)
            
            plotter.plot_data(dataset_name=ep_prefix+"TrackingError"+dset_suffix, 
                title=ep_prefix+"TrackingError"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                ylabel="m/s",
                use_markers=True,
                marker_size=marker_size,
                data_labels=track_err_names,
                data_idxs=None)

            patterns=["*z_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, contact_forces_names)
            plotter.plot_data(dataset_name=ep_prefix+"RhcContactForces"+dset_suffix, 
                title=ep_prefix+"Vertical MPC contact f (base loc)"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=False,
                marker_size=marker_size,
                data_labels=selected,
                data_idxs=idxs)
            
            plotter.plot_data(dataset_name=ep_prefix+"RhcFailIdx"+dset_suffix, 
                title=ep_prefix+"Rhc fail idx"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels="fail_idx",
                data_idxs=None)
            
            plotter.plot_data(dataset_name=ep_prefix+"RhcRefsFlag"+dset_suffix, 
                title=ep_prefix+"Rhc refs flags"+dset_suffix, 
                xaxis_dataset_name=xaxis_dataset_name,
                xlabel=xlabel,
                use_markers=True,
                marker_size=marker_size,
                data_labels=rhc_refs_names,
                data_idxs=None)
             
            # plotting contact phases
            from lrhc_control.utils.data_postproc.contact_visual import ContactPlotter
            patterns=["fc_contact*z*"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            vertical_contact_f=plotter.data[obs_datasetname][:, :, idxs]
            valid_mask = np.isfinite(vertical_contact_f[:, 0, 0])
            patterns=["linvel_*_ref_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            linvel_ref=plotter.data[obs_datasetname][:, 0, idxs][valid_mask, :]
            patterns=["linvel_x_base_loc", "linvel_y_base_loc", "linvel_z_base_loc"]
            idxs,selected=plotter.get_idx_matching(patterns, obs_names)
            linvel_meas=plotter.data[obs_datasetname][:, 0, idxs][valid_mask, :]
            valid_f=vertical_contact_f[valid_mask, 0, :]
            is_contact=valid_f>=1e-3
            contact_state=np.full_like(valid_f, fill_value=0.0)
            contact_state[is_contact]=1.0
            contact_plotter=ContactPlotter(data=contact_state.T,
                ref_vel=linvel_ref.T,meas_vel=linvel_meas.T)
            contact_plotter.plot()

    plotter.show() # Display all plots


    

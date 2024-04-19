from control_cluster_bridge.controllers.rhc import RHController

from lrhc_control.controllers.rhc.horizon_based.horizon_imports import * 

from lrhc_control.controllers.rhc.horizon_based.hybrid_quad_rhc_refs import HybridQuadRhcRefs
from lrhc_control.controllers.rhc.horizon_based.gait_manager import GaitManager

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import Journal, LogType

import numpy as np

import os
import shutil

import time

class HybridQuadRhc(RHController):

    def __init__(self, 
            srdf_path: str,
            urdf_path: str,
            config_path: str,
            robot_name: str, # used for shared memory namespaces
            codegen_dir: str, 
            n_nodes:float = 25,
            dt: float = 0.02,
            max_solver_iter = 1, # defaults to rt-iteration
            open_loop: bool = True,
            dtype = np.float32,
            verbose = False, 
            debug = False
            ):

        self._open_loop = open_loop

        self._codegen_dir = codegen_dir
        if not os.path.exists(self._codegen_dir):
            os.makedirs(self._codegen_dir)
        # else:
        #     # Directory already exists, delete it and recreate
        #     shutil.rmtree(self._codegen_dir)
        #     os.makedirs(self._codegen_dir)

        self.step_counter = 0
        self.sol_counter = 0
    
        self.max_solver_iter = max_solver_iter
        
        self._timer_start = time.perf_counter()
        self._prb_update_time = time.perf_counter()
        self._phase_shift_time = time.perf_counter()
        self._task_ref_update_time = time.perf_counter()
        self._rti_time = time.perf_counter()

        self.robot_name = robot_name
        
        self.config_path = config_path

        self.urdf_path = urdf_path
        # read urdf and srdf files
        with open(self.urdf_path, 'r') as file:
            self.urdf = file.read()
        self._base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        super().__init__(srdf_path = srdf_path,
                        n_nodes = n_nodes,
                        dt = dt,
                        namespace = self.robot_name,
                        dtype = dtype,
                        verbose = verbose, 
                        debug = debug)

        self.rhc_costs={}
        self.rhc_constr={}
    
    def _get_quat_remap(self):
        # overrides parent
        return [1, 2, 3, 0] # mapping from robot quat. to Horizon's quaternion convention
    
    def _init_problem(self):
        
        self.urdf = self.urdf.replace('continuous', 'revolute') # continous joint is parametrized
        # in So2, so will add 

        self._kin_dyn = casadi_kin_dyn.CasadiKinDyn(self.urdf)

        self._assign_controller_side_jnt_names(jnt_names=self._get_robot_jnt_names())

        self._prb = Problem(self._n_intervals, 
                        receding=True, 
                        casadi_type=cs.SX)
        self._prb.setDt(self._dt)

        self._init_robot_homer()

        init = self._base_init.tolist() + list(self._homer.get_homing())

        FK = self._kin_dyn.fk('ball_1') # just to get robot reference height
        
        kyon_wheel_radius = 0.124 # hardcoded!!!!

        init_pos_foot = FK(q=init)['ee_pos']

        self._base_init[2] = -init_pos_foot[2] + kyon_wheel_radius # override init

        self._model = FullModelInverseDynamics(problem=self._prb,
                                kd=self._kin_dyn,
                                q_init=self._homer.get_homing_map(),
                                base_init=self._base_init)
        
        self._ti = TaskInterface(prb=self._prb, 
                            model=self._model, 
                            max_solver_iter=self.max_solver_iter,
                            debug = self._debug,
                            verbose = self._verbose, 
                            codegen_workdir = self._codegen_dir)
        
        self._ti.setTaskFromYaml(self.config_path)

        # setting initial base pos ref
        base_pos = self._ti.getTask('base_position')

        base_pos.setRef(np.atleast_2d(self._base_init).T)

        self._tg = trajectoryGenerator.TrajectoryGenerator()

        self._pm = pymanager.PhaseManager(self._n_nodes)

        # adding timelines
        self._init_contact_timelines()

        self._ti.model.q.setBounds(self._ti.model.q0, self._ti.model.q0, nodes=0)
        self._ti.model.v.setBounds(self._ti.model.v0, self._ti.model.v0, nodes=0)
        # ti.model.a.setBounds(np.zeros([model.a.shape[0], 1]), np.zeros([model.a.shape[0], 1]), nodes=0)
        self._ti.model.q.setInitialGuess(self._ti.model.q0)
        self._ti.model.v.setInitialGuess(self._ti.model.v0)

        f0 = [0, 0, self._kin_dyn.mass() / 4 * 9.8]
        for _, cforces in self._ti.model.cmap.items():
            for c in cforces:
                c.setInitialGuess(f0)

        self._ti.finalize()

        self._ti.bootstrap()

        self._ti.init_inv_dyn_for_res() # we initialize some objects for sol. postprocessing purposes

        self._ti.load_initial_guess()

        contact_phase_map = {c: f'{c}_timeline' for c in self._model.cmap.keys()}
        
        self._gm = GaitManager(self._ti, self._pm, contact_phase_map)

        self.n_dofs = self._get_ndofs() # after loading the URDF and creating the controller we
        # know n_dofs -> we assign it (by default = None)

        self.n_contacts = len(self._model.cmap.keys())
        
        # self.horizon_anal = analyzer.ProblemAnalyzer(self._prb)
    
    def _init_contact_timelines(self):

        c_phases = dict()
        for c in self._model.cmap.keys():
            c_phases[c] = self._pm.addTimeline(f'{c}_timeline')

        stance_duration_default = 5
        flight_duration_default = 5

        for c in self._model.cmap.keys():

            # stance phase normal
            stance_phase = pyphase.Phase(stance_duration_default, f'stance_{c}')

            # add contact constraints to phase
            if self._ti.getTask(f'{c}_contact') is not None:

                stance_phase.addItem(self._ti.getTask(f'{c}_contact'))

            else:

                raise Exception(f"[{self.__class__.__name__}]" + \
                                f"[{self.journal.exception}]" + \
                                ": task not found")

            # register phase to timeline
            c_phases[c].registerPhase(stance_phase)

            # flight phase normal
            flight_phase = pyphase.Phase(flight_duration_default, f'flight_{c}')

            init_z_foot = self._model.kd.fk(c)(q=self._model.q0)['ee_pos'].elements()[2]

            ref_trj = np.zeros(shape=[7, flight_duration_default])
            
            # trajectory on z
            ref_trj[2, :] = np.atleast_2d(self._tg.from_derivatives(flight_duration_default, 
                                                        init_z_foot, 
                                                        init_z_foot, 
                                                        0.1, 
                                                        [0, 0, 0]))
            
            if self._ti.getTask(f'z_{c}') is not None:

                flight_phase.addItemReference(self._ti.getTask(f'z_{c}'), ref_trj)
                
            else:
                 
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + f": task {c}_contact not found")
            
            # flight_phase.addConstraint(prb.getConstraints(f'{c}_vert'), nodes=[0 ,flight_duration-1])  # nodes=[0, 1, 2]
            c_phases[c].registerPhase(flight_phase)

        
        for c in self._model.cmap.keys():
            stance = c_phases[c].getRegisteredPhase(f'stance_{c}')
            # flight = c_phases[c].getRegisteredPhase(f'flight_{c}')
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)
            c_phases[c].addPhase(stance)

    def _init_rhc_task_cmds(self):
        
        rhc_refs = HybridQuadRhcRefs(gait_manager=self._gm,
                    robot_index=self.controller_index,
                    namespace=self.namespace,
                    safe=False, 
                    verbose=self._verbose,
                    vlevel=VLevel.V2)
        
        rhc_refs.run()

        rhc_refs.rob_refs.set_jnts_remapping(jnts_remapping=self._to_controller)
        rhc_refs.rob_refs.set_q_remapping(q_remapping=self._get_quat_remap())
              
        # writing initializations
        rhc_refs.reset(p_ref=np.atleast_2d(self._base_init)[:, 0:3], 
            q_ref=np.atleast_2d(self._base_init)[:, 3:7] # will be remapped according to just set q_remapping
            )
        
        return rhc_refs
    
    def _get_contact_names(self):

        return list(self._ti.model.cmap.keys())
    
    def _get_robot_jnt_names(self):

        joints_names = self._kin_dyn.joint_names()

        to_be_removed = ["universe", 
                        "reference", 
                        "world", 
                        "floating", 
                        "floating_base"]
        
        for name in to_be_removed:

            if name in joints_names:
                joints_names.remove(name)

        return joints_names
    
    def _get_ndofs(self):
        
        return len(self._model.joint_names)

    def _get_cmd_jnt_q_from_sol(self):
        
        # wrapping joint q commands between 2pi and -2pi
        # (to be done for the simulator)
        return np.fmod(self._ti.solution['q'][7:, 1], 2*np.pi).reshape(1,  
                                        self.robot_cmds.n_jnts())
    
    def _get_cmd_jnt_v_from_sol(self):

        return self._ti.solution['v'][6:, 1].reshape(1,  
                                        self.robot_cmds.n_jnts())

    def _get_cmd_jnt_eff_from_sol(self):
        
        efforts_on_first_node = self._ti.eval_efforts_on_first_node()

        return efforts_on_first_node[6:, 0].reshape(1,  
                                        self.robot_cmds.n_jnts())
    
    def _get_rhc_cost(self):

        return self._ti.solution["opt_cost"]
    
    def _get_rhc_constr_viol(self):

        return self._ti.solution["residual_norm"]
    
    def _get_rhc_nodes_cost(self):

        cost = self._ti.solver_rti.getCostValOnNodes()
        return cost.reshape((1, -1))
    
    def _get_rhc_nodes_constr_viol(self):
        
        constr_viol = self._ti.solver_rti.getConstrValOnNodes()
        return constr_viol.reshape((1, -1))
    
    def _get_rhc_niter_to_sol(self):

        return self._ti.solution["n_iter2sol"]
    
    def _assemble_meas_robot_state(self):
        
        p = self.robot_state.root_state.get(data_type="p", robot_idxs=self.controller_index).reshape(-1, 1)
        q_root = self.robot_state.root_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)
        q_jnts = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)
        v_root = self.robot_state.root_state.get(data_type="v", robot_idxs=self.controller_index).reshape(-1, 1)
        omega = self.robot_state.root_state.get(data_type="omega", robot_idxs=self.controller_index).reshape(-1, 1)
        v_jnts = self.robot_state.jnts_state.get(data_type="v", robot_idxs=self.controller_index).reshape(-1, 1)

        return np.concatenate((p, q_root, q_jnts, v_root, omega, v_jnts),
                axis=0)

    def _assemble_meas_robot_configuration(self):
        
        p = self.robot_state.root_state.get(data_type="p", robot_idxs=self.controller_index).reshape(-1, 1)
        q_root = self.robot_state.root_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)
        q_jnts = self.robot_state.jnts_state.get(data_type="q", robot_idxs=self.controller_index).reshape(-1, 1)

        return np.concatenate((p, q_root, q_jnts),
                axis=10)
    
    def _update_open_loop(self):

        shift_num = -1 # shift data by one node

        # building ig for state
        xig = np.roll(self._ti.solution['x_opt'], 
                shift_num, axis=1) # rolling state sol.
        for i in range(abs(shift_num)):
            # state on last node is copied to the elements
            # which are "lost" during the shift operation
            xig[:, -1 - i] = self._ti.solution['x_opt'][:, -1]

        # building ig for inputs
        uig = np.roll(self._ti.solution['u_opt'], 
                shift_num, axis=1) # rolling state sol.
        for i in range(abs(shift_num)):
            # state on last node is copied to the elements
            # which are "lost" during the shift operation
            uig[:, -1 - i] = self._ti.solution['u_opt'][:, -1]

        # assigning ig
        self._prb.getState().setInitialGuess(xig)
        self._prb.getInput().setInitialGuess(uig)

        # open loop update:
        # state on first node is second state 
        # shifted
        self._prb.setInitialState(x0=xig[:, 0])
    
    def _update_closed_loop(self):

        shift_num = -1 # shift data by one node
        
        # building ig for state
        xig = np.roll(self._ti.solution['x_opt'], 
                shift_num, axis=1) # rolling state sol.
        for i in range(abs(shift_num)):
            # state on last node is copied to the elements
            # which are "lost" during the shift operation
            xig[:, -1 - i] = self._ti.solution['x_opt'][:, -1]

        # building ig for inputs
        uig = np.roll(self._ti.solution['u_opt'], 
                shift_num, axis=1) # rolling state sol.
        for i in range(abs(shift_num)):
            # state on last node is copied to the elements
            # which are "lost" during the shift operation
            uig[:, -1 - i] = self._ti.solution['u_opt'][:, -1]

        # assigning ig
        self._prb.getState().setInitialGuess(xig)
        self._prb.getInput().setInitialGuess(uig)

        # sets state on node 0 from measurements
        robot_state = self._assemble_meas_robot_state()
        self._prb.setInitialState(x0=
                        robot_state)
    
    def _solve(self):
        
        if self._debug:
            return self._db_solve()
        else:
            return self._min_solve()
        
    def _min_solve(self):
        # minimal solve version -> no debug 
        if self._open_loop:
            self._update_open_loop() # updates the TO ig and 
            # initial conditions using data from the solution itself
        else: 
            self._update_closed_loop() # updates the TO ig and 
            # initial conditions using robot measurements
    
        self._pm.shift() # shifts phases of one dt
        self.rhc_refs.step() # updates rhc references
        # with the latests available data on shared memory
            
        try:
            converged = self._ti.rti() # solves the problem
            self.sol_counter = self.sol_counter + 1
            return True
        except Exception as e:
            return False
    
    def _db_solve(self):

        self._timer_start = time.perf_counter()

        if self._open_loop:
            self._update_open_loop() # updates the TO ig and 
            # initial conditions using data from the solution itself
        else: 
            self._update_closed_loop() # updates the TO ig and 
            # initial conditions using robot measurements
        
        self._prb_update_time = time.perf_counter() 
        self._pm.shift() # shifts phases of one dt
        self._phase_shift_time = time.perf_counter() 
        self.rhc_refs.step() # updates rhc references
        self._task_ref_update_time = time.perf_counter() 
            
        try:
            converged = self._ti.rti() # solves the problem
            self._rti_time = time.perf_counter() 
            self.sol_counter = self.sol_counter + 1
            self._update_db_data()
            return True
        except Exception as e:
            if self._verbose:
                exception = f"Rti() for controller {self.controller_index} failed" + \
                f" with exception{type(e).__name__}"
                Journal.log(self.__class__.__name__,
                    "solve",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = False)
            self._update_db_data()
            return False
        
    def _update_db_data(self):

        self._profiling_data_dict["problem_update_dt"] = self._prb_update_time - self._timer_start
        self._profiling_data_dict["phases_shift_dt"] = self._phase_shift_time - self._prb_update_time
        self._profiling_data_dict["task_ref_update"] = self._task_ref_update_time - self._phase_shift_time
        self._profiling_data_dict["rti_solve_dt"] = self._rti_time - self._task_ref_update_time
        self.rhc_costs.update(self._ti.solver_rti.getCostsValues())
        self.rhc_constr.update(self._ti.solver_rti.getConstraintsValues())

    def _reset(self):
        
        # reset task interface (ig, solvers, etc..) + 
        # phase manager
        self._gm.reset()
        # we also re-initialize contact timelines
        # self._init_contact_timelines()
        # resets rhc references
        if self.rhc_refs is not None:
            self.rhc_refs.reset(p_ref=np.atleast_2d(self._base_init)[:, 0:3], 
                        q_ref=np.atleast_2d(self._base_init)[:, 3:7]
                        )

    def _get_cost_data(self):
        
        cost_dict = self._ti.solver_rti.getCostsValues()
        cost_names = list(cost_dict.keys())
        cost_dims = [1] * len(cost_names) # costs are always scalar
        return cost_names, cost_dims
    
    def _get_constr_data(self):
        
        constr_dict = self._ti.solver_rti.getConstraintsValues()
        constr_names = list(constr_dict.keys())
        constr_dims = [-1] * len(constr_names)
        i = 0
        for constr in constr_dict:
            constr_val = constr_dict[constr]
            constr_shape = constr_val.shape
            constr_dims[i] = constr_shape[0]
            i+=1
        return constr_names, constr_dims
    
    def _get_q_from_sol(self):
        return self._ti.solution['q']

    def _get_v_from_sol(self):
        return self._ti.solution['v']
    
    def _get_a_from_sol(self):
        return self._ti.solution['a']
    
    def _get_a_dot_from_sol(self):
        return None
    
    def _get_f_from_sol(self):
        # to be overridden by child class
        contact_names =self._get_contacts() # we use controller-side names
        try: 
            data = [self._ti.solution["f_" + key] for key in contact_names]
            return np.concatenate(data, axis=0)
        except:
            return None
        return None
    
    def _get_f_dot_from_sol(self):
        # to be overridden by child class
        return None
    
    def _get_eff_from_sol(self):
        # to be overridden by child class
        return None
    
    def _get_cost_from_sol(self,
                    cost_name: str):
        return self.rhc_costs[cost_name]
    
    def _get_constr_from_sol(self,
                    constr_name: str):
        return self.rhc_constr[constr_name]
from aug_mpc.utils.shared_data.agent_refs import AgentRefs

from mpc_hive.utilities.shared_data.rhc_data import RobotState
from mpc_hive.utilities.math_utils import world2base_frame_twist

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

import math
import time
import numpy as np

class AgentRefsFromJoy:

    def __init__(self, 
                namespace: str, 
                verbose = False,
                agent_refs_world: bool = True,
                env_idx: int = None,
                hold_time: float = 0.01):   # <-- new hold_time parameter
        self._env_idx=env_idx

        self._verbose = verbose

        self._agent_refs_world=agent_refs_world
        
        self.namespace = namespace

        self._closed = False
        
        self.enable_linvel = False
        self.enable_omega = False
        self.enable_omega_roll = False
        self.enable_omega_pitch = False
        self.enable_omega_yaw = False

        self.enable_pos = False

        self.dpos = 0.1 # [m]
        self.dxy = 0.05 # [m/s]
        self.dvxyz = 0.05 # [m/s]
        self.dheading=0.05
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self._v_magnitude=0.0
        self._heading_lat=0.0
        self._heading_frontal=0.0
        self._heading=0.0
        self.agent_refs = None

        self._max_vxy_magn=0.8 # [m/s]
        self._max_vz_magn=0.0
        self._max_pitch_rate=0.0 # [rad/s]
        self._max_roll_rate=0.0 # [rad/s]
        self._max_yaw_rate=0.4 # [rad/s]

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._twist_null = None

        # hold time for toggles (seconds)
        self.hold_time = float(hold_time)

        # helper structures to manage press-and-hold toggles
        # keys: "omega", "linvel", "pos"
        self._hold_since = {
            "omega": None,
            "linvel": None,
            "pos": None
        }
        # once a toggle has fired for the current press, mark True until release
        self._hold_triggered = {
            "omega": False,
            "linvel": False,
            "pos": False
        }

        self._init_shared_data()

    def _init_shared_data(self):
        
        self.env_index=None
        if self._env_idx is None:
            self.env_index = SharedTWrapper(namespace = self.namespace,
                    basename = "EnvSelector",
                    is_server = False, 
                    verbose = True, 
                    vlevel = VLevel.V2,
                    safe = False,
                    dtype=dtype.Int)
            
            self.env_index.run()
        
        self._init_rhc_ref_subscriber()
        
        self._current_twist_ref_world = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        self._current_twist_ref_base=np.full_like(self._current_twist_ref_world, fill_value=0.0).reshape(1, -1)

        self._current_pos_ref = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        
        self._robot_state = RobotState(namespace=self.namespace,
                            is_server=False, 
                            safe=False,
                            verbose=True,
                            vlevel=VLevel.V2)
        self._robot_state.run()            

    def _init_rhc_ref_subscriber(self):

        self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False, 
                                safe=True, 
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                with_gpu_mirror=False,
                                with_torch_view=False)

        self.agent_refs.run()

        self._twist_null = self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np)
        self._twist_null[:]=0.0

        q0=np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="q", robot_idxs=self.cluster_idx_np),fill_value=0.0)
        q0[0]=1.0
        self.agent_refs.rob_refs.root_state.set(data_type="q",data=q0,
                                        robot_idxs=self.cluster_idx_np)
        
    def __del__(self):

        if not self._closed:
            self._close()
    
    def _close(self):
        
        if self.agent_refs is not None:
            self.agent_refs.close()
        if self._robot_state is not None:
            self._robot_state.close()

        self._closed = True
    
    def _synch(self, joy):
        
        if self.env_index is not None:
            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self._env_idx=env_index[0, 0].item()
        self.cluster_idx = self._env_idx
        self.cluster_idx_np = self.cluster_idx    
        
        # Check hold-and-toggle for toggles:
        # face[1] -> omega toggle, face[0] -> linvel toggle, face[2] -> pos toggle
        self._check_and_toggle("omega", bool(getattr(joy, "face")[1] if hasattr(joy, "face") else False))
        self._check_and_toggle("linvel", bool(getattr(joy, "face")[0] if hasattr(joy, "face") else False))
        self._check_and_toggle("pos", bool(getattr(joy, "face")[2] if hasattr(joy, "face") else False))

        # After managing toggles, update twist/pos using current stable flags & latest joy values
        self._set_omega(joy)    
        self._set_linvel(joy)    
        self._set_position(joy)     

    def _check_and_toggle(self, name: str, pressed: bool):
        """
        Generic helper to toggle a named flag only when the corresponding 'pressed' boolean
        has been True for at least self.hold_time seconds.

        name: one of "omega", "linvel", "pos"
        pressed: current boolean from joy
        """
        now = time.time()
        # ensure name valid
        if name not in self._hold_since:
            return

        if pressed:
            if self._hold_since[name] is None:
                # press started
                self._hold_since[name] = now
                # do not toggle yet; wait for hold_time
            else:
                # already pressing; check duration
                duration = now - self._hold_since[name]
                if duration >= self.hold_time and not self._hold_triggered[name]:
                    # perform the toggle
                    if name == "omega":
                        self.enable_omega = not self.enable_omega
                        info = f"Twist change enabled: {self.enable_omega}"
                        Journal.log(self.__class__.__name__,
                            "_set_omega",
                            info,
                            LogType.INFO,
                            throw_when_excep = True)
                    elif name == "linvel":
                        self.enable_linvel = not self.enable_linvel
                        info = f"High level navigation enabled: {self.enable_linvel}"
                        Journal.log(self.__class__.__name__,
                            "_set_linvel",
                            info,
                            LogType.INFO,
                            throw_when_excep = True)
                    elif name == "pos":
                        self.enable_pos = not self.enable_pos
                        info = f"High level pos reference change: {self.enable_pos}"
                        Journal.log(self.__class__.__name__,
                            "_set_position",
                            info,
                            LogType.INFO,
                            throw_when_excep = True)
                    # mark triggered until release to avoid repeated toggles for a single hold
                    self._hold_triggered[name] = True
        else:
            # not pressed: clear hold start and triggered flag so next press can toggle again
            self._hold_since[name] = None
            self._hold_triggered[name] = False

    # def _set_omega(self, joy):
    #     """
    #     Set angular part of world twist based on joystick inputs.

    #     - roll and pitch rates are forced to 0.0 always.
    #     - yaw sign is chosen by hat up/down (joy.hat[1]): up -> +, down -> -, neutral -> 0.
    #     - magnitude is taken from right trigger (joy.triggers[1]) and mapped to [0,1].
    #     """
    #     # If omega not enabled -> zero angular components
    #     if not self.enable_omega:
    #         self._current_twist_ref_world[3:] = 0.0
    #         return

    #     # Ensure roll & pitch are zero as requested
    #     self._current_twist_ref_world[3] = 0.0  # roll rate
    #     self._current_twist_ref_world[4] = 0.0  # pitch rate

    #     # Read hat Y for direction: joy.hat is [x, y], value in {-1,0,1}
    #     try:
    #         hat_y = int(joy.hat[1])
    #     except Exception:
    #         hat_y = 0

    #     # Determine sign from hat: up (1) -> +1, down (-1) -> -1, neutral -> 0
    #     if hat_y > 0:
    #         yaw_sign = 1.0
    #     elif hat_y < 0:
    #         yaw_sign = -1.0
    #     else:
    #         yaw_sign = 0.0

    #     # Read right trigger magnitude (index 1)
    #     try:
    #         rt = float(joy.triggers[1])
    #     except Exception:
    #         rt = 0.0

    #     # Normalize trigger to [0,1].
    #     # Some platforms give triggers in [-1,1] (rest ~ -1, pressed ~ +1),
    #     # others give [0,1] (rest ~ 0). Detect and normalize:
    #     if rt < -0.1:
    #         # assume [-1,1] range -> map to [0,1]
    #         norm_rt = (rt + 1.0) / 2.0
    #     else:
    #         # assume [0,1] or small positive noise
    #         norm_rt = rt

    #     # clamp and deadzone
    #     norm_rt = float(np.clip(norm_rt, 0.0, 1.0))
    #     deadzone = 1e-3
    #     if norm_rt < deadzone or yaw_sign == 0.0:
    #         yaw_rate = 0.0
    #     else:
    #         yaw_rate = yaw_sign * norm_rt * float(self._max_yaw_rate)

    #     # clip final yaw to allowed bounds (safety)
    #     yaw_rate = float(np.clip(yaw_rate, -self._max_yaw_rate, self._max_yaw_rate))

    #     # write into world twist angular part [roll,pitch,yaw] -> indices 3,4,5
    #     self._current_twist_ref_world[3] = 0.0
    #     self._current_twist_ref_world[4] = 0.0
    #     self._current_twist_ref_world[5] = yaw_rate

    def _set_omega(self, joy):
        """
        Set angular part of world twist based on the left stick horizontal axis.

        - roll and pitch are always zero.
        - yaw sign/magnitude derived from left stick X (joy.sticks[0]).
        right -> +, left -> -.
        - Uses a small deadzone (self.dxy) to avoid tiny jitter.
        - Final yaw rate clipped to [-self._max_yaw_rate, self._max_yaw_rate].
        """
        twist_ref = self._current_twist_ref_world
        # If omega not enabled -> zero angular components
        if not self.enable_omega:
            twist_ref[3:] = 0.0
            return

        # Ensure roll & pitch are zero as requested
        twist_ref[3] = 0.0  # roll rate
        twist_ref[4] = 0.0  # pitch rate

        # read left stick horizontal axis (left_x)
        try:
            lx = float(-joy.sticks[0])
        except Exception:
            lx = 0.0

        # deadzone to avoid noise around center
        deadzone = float(getattr(self, "dxy", 0.05))
        if abs(lx) <= deadzone:
            yaw_rate = 0.0
        else:
            # normalize magnitude: use absolute stick displacement (assume stick in [-1,1])
            mag = min(abs(lx), 1.0)
            # linear mapping; you can change to mag**2 for non-linear response
            yaw_rate = np.sign(lx) * mag * float(self._max_yaw_rate)

        # write into world twist angular part [roll,pitch,yaw] -> indices 3,4,5
        twist_ref[5] = yaw_rate

        twist_ref[3] = np.clip(twist_ref[3], a_min=-self._max_roll_rate, a_max=self._max_roll_rate)
        twist_ref[4] = np.clip(twist_ref[4], a_min=-self._max_pitch_rate, a_max=self._max_pitch_rate)
        twist_ref[5] = np.clip(twist_ref[5], a_min=-self._max_yaw_rate, a_max=self._max_yaw_rate)
        
    def _set_linvel(self, joy):
        if not self.enable_linvel:
            self._current_twist_ref_world[0:3] = 0.0
        else:
            twist_ref = self._current_twist_ref_world

            # Read left stick from the provided joy object
            # Expected layout: sticks = [left_x, left_y, right_x, right_y]
            try:
                lx = float(joy.sticks[2])
                ly = float(joy.sticks[3])
            except Exception:
                lx, ly = 0.0, 0.0

            # Compute magnitude and apply deadzone
            mag = float(np.hypot(lx, ly))

            if mag < self.dxy:
                # near center: stop translational motion but keep heading stable
                self._v_magnitude = 0.0
                # do not update self._heading (preserve previous heading)
            else:
                # update heading (atan2(y, x))
                self._heading = np.arctan2(ly, lx)-math.pi/2.0

                # Normalize magnitude: joystick typically in [-1,1], hypot max is sqrt(2)
                # We clamp to 1.0 to be conservative; optionally divide by sqrt(2) if you want full-range normalization
                norm_mag = min(mag, 1.0)

                # Linear mapping to speed; you can replace with a non-linear curve if desired
                self._v_magnitude = norm_mag * self._max_vxy_magn

            # clamp and write into twist_ref
            self._v_magnitude = np.clip(self._v_magnitude, a_min=0.0, a_max=self._max_vxy_magn)
            twist_ref[0] = self._v_magnitude * np.cos(self._heading)
            twist_ref[1] = self._v_magnitude * np.sin(self._heading)

            # vertical velocity: not supported by agent
            twist_ref[2] = 0.0

            # angular rates: clip as before
            
    def _set_position(self, joy):
        """
        Incrementally update self._current_pos_ref[0:2] using the right stick (sticks[2], sticks[3]).
        - When enable_pos is False: reset current pos ref to robot's current position (as before).
        - When enable_pos is True: disable linear velocity, and increment position by v * dt,
        where v = stick_value * self._max_vxy_magn and dt is time since last update.
        - Z (index 2) is left unchanged.
        """
        now = time.time()

        # If enable_pos toggled off -> reset position to robot's current position (same behavior you had)
        if not self.enable_pos:
            # reset
            try:
                robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
                robot_p[2] = 0.0
                self._current_pos_ref[:] = robot_p
            except Exception:
                # fallback: do nothing if we can't read robot state
                pass

            # clear last update timestamp so next enable starts fresh
            if hasattr(self, "_last_pos_update_time"):
                delattr(self, "_last_pos_update_time")
            return

        # If we are here, enable_pos is True
        # Ensure linear velocity is disabled while position control is active
        if self.enable_linvel:
            self.enable_linvel = False
            # zero linear twist components to avoid conflicts
            try:
                self._current_twist_ref_world[0:3] = 0.0
            except Exception:
                pass

        # read right stick (expected layout: sticks = [left_x,left_y,right_x,right_y])
        try:
            rx = float(joy.sticks[3]) 
            ry = -float(joy.sticks[2])
        except Exception:
            rx, ry = 0.0, 0.0

        # deadzone: ignore small noise near center
        if np.hypot(rx, ry) < self.dxy:
            # no change to target position
            # update last timestamp so dt doesn't accumulate large value next time
            self._last_pos_update_time = now
            return

        # determine dt since last update (safety: clamp dt to a sane maximum)
        last = getattr(self, "_last_pos_update_time", None)
        if last is None:
            dt = 0.0
        else:
            dt = now - last
        # avoid huge dt (e.g., if paused); cap to 0.1s so a long pause won't teleport target
        dt = float(np.clip(dt, 0.0, 0.1))
        # store timestamp for next round
        self._last_pos_update_time = now

        if dt <= 0.0:
            # nothing to integrate yet (first call after enabling)
            return

        # compute desired velocity in world frame from stick
        # stick in [-1,1], so full deflection -> max velocity self._max_vxy_magn (m/s)
        vx = np.clip(rx, -1.0, 1.0) * float(self._max_vxy_magn)
        vy = np.clip(ry, -1.0, 1.0) * float(self._max_vxy_magn)

        # delta position = v * dt
        dx = vx * dt
        dy = vy * dt

        # apply delta to current reference (world frame)
        try:
            # Ensure _current_pos_ref exists and is length >= 2
            if self._current_pos_ref is None or len(self._current_pos_ref) < 2:
                # attempt to initialize from robot state
                try:
                    robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
                    robot_p[2] = 0.0
                    self._current_pos_ref = robot_p
                except Exception:
                    # give up if we can't
                    return

            # Increment x,y. Note: user requested z should not change.
            self._current_pos_ref[0] += dx
            self._current_pos_ref[1] += dy

            # Optionally clamp huge jumps (defensive): limit per-call displacement by dpos if desired
            # If you prefer a fixed stepping (dpos) instead of velocity scaling, replace above with:
            #   self._current_pos_ref[0] += np.clip(rx, -1, 1) * self.dpos
            #   self._current_pos_ref[1] += np.clip(ry, -1, 1) * self.dpos

            # Ensure z unchanged
            if len(self._current_pos_ref) > 2:
                # keep whatever z was (or zero)
                self._current_pos_ref[2] = float(self._current_pos_ref[2])
        except Exception:
            # swallow exceptions to keep loop robust
            pass

    def _write_to_shared_mem(self):

        self.agent_refs.rob_refs.root_state.synch_all(read=True)
        self._robot_state.root_state.synch_all(read = True, retry = True) # read robot state        
        
        if self.enable_pos:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2]=0.0
            # self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref-robot_p,
            #                                 robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref,
                                            robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                        n_rows=1, n_cols=3,
                                        read=False)
            
        if self._agent_refs_world:
            
            robot_q = self._robot_state.root_state.get(data_type="q")[self.cluster_idx_np, :].reshape(1, -1)

            if self.enable_omega:
                # ref was set in world frame -> we need to move it in base frame before setting it to the agent
                
                # rotate only omega
                world2base_frame_twist(t_w=self._current_twist_ref_world.reshape(1, -1), 
                    q_b=robot_q, 
                    t_out=self._current_twist_ref_base,
                    omega=True, # keep omega ref in world frame
                    linvel=False # linvel in base
                    )
            if self.enable_linvel:
                world2base_frame_twist(t_w=self._current_twist_ref_world.reshape(1, -1), 
                    q_b=robot_q, 
                    t_out=self._current_twist_ref_base,
                    omega=False, # keep omega ref in world frame
                    linvel=True # linvel in base
                    )
                # self._current_twist_ref_base[:, 0:3]=self._current_twist_ref_world.reshape(1, -1)[:, 0:3]

        else:
            self._current_twist_ref_base[:, :]=self._current_twist_ref_world.reshape(1, -1)

        self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._current_twist_ref_base,
                                        robot_idxs=self.cluster_idx_np)
        self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=7, 
                                    n_rows=1, n_cols=6,
                                    read=False)
              
    def run(self, connect, topic, poll_interval ):

        info = f"Ready. Starting to listen for commands..."

        def on_message(payload):
            pass
        
        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
                
        import time

        self.agent_refs.run()
        
        from mpc_hive.utilities.joy.joy_zmq_listener import JoyListenerZMQ

        joy_listener=JoyListenerZMQ(connect=connect, topic=topic, poll_interval=poll_interval, on_message=on_message)
        joy_listener.start()

        try:
            while not joy_listener.done:
                self._synch(joy_listener) # set refs
                self._write_to_shared_mem() # write refs
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("[AgentRefsFromJoy][run]: Exiting...")
        finally:
            joy_listener.stop()


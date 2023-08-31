# TODO DOCUMNET & CLEAN

import torch 
import zonopy as zp
import matplotlib.pyplot as plt 
#from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d as a3
import os
def wrap_to_pi(phases):
    return (phases + torch.pi) % (2 * torch.pi) - torch.pi

T_PLAN, T_FULL = 0.5, 1

from urchin import URDF
# robots2.DEBUG_VIZ = False
from typing import Union, Tuple

import numpy as np
import trimesh
from collections import OrderedDict
import copy

from enum import Enum
class STEP_TYPE(str, Enum):
    DIRECT = 'direct'
    INTEGRATION = 'integration'

class RENDERER(str, Enum):
    PYRENDER = 'pyrender'
    PYRENDER_OFFSCREEN = 'pyrender-offscreen'
    # MATPLOTLIB = 'matplotlib' # disabled for now

# A base environment for basic kinematic simulation of any URDF loaded by Urchin without any reward
class KinematicUrdfBase:
    # STEP_TYPE = ['direct', 'integration']
    # RENDERER = ['pyrender', 'pyrender-offscreen', 'matplotlib']
    DIMENSION = 3

    def __init__(self,
            robot: Union[URDF, str] = 'Kinova3', # robot model
            max_episode_steps: int = 300,
            step_type: STEP_TYPE = 'direct',
            t_step: float = 0.5,
            timestep_discretization: int = 50,       # these are values per t_step
            integration_discretization: int = 100,   # these are values per t_step, if step type is direct, this is ignored.
            check_collision: bool = True,
            check_self_collision: bool = False,
            verbose_self_collision: bool = False,
            use_bb_collision: bool = True,
            collision_discretization: int = 500,     # these are values per t_step, if self_collision is False, this is ignored.
            collision_use_visual: bool = True,
            check_joint_limits: bool = True,
            enforce_joint_pos_limit: bool = True,
            verbose_joint_limits: bool = False,
            renderer: RENDERER = 'pyrender',
            render_frames: int = 30,
            render_fps: int = 60,
            render_size: Tuple[int, int] = (600, 600),
            render_mesh: bool = True,
            renderer_kwargs: dict = {},
            reopen_on_close: bool = True,
            seed: int = 42,
            ):
        
        # Setup the random number generator
        self.np_random = np.random.default_rng(seed=seed)

        ### Robot Specs
        # Load the robot & get important properties
        if isinstance(robot, str):
            robot = URDF.load(robot)
        self.robot = robot
        self.robot_graph = self.robot._G.to_undirected()
        
        self.dof = len(self.robot.actuated_joints)
        self.continuous_joints = []
        pos_lim = [[-np.Inf, np.Inf]]*self.dof
        vel_lim = [np.Inf]*self.dof
        eff_lim = [np.Inf]*self.dof
        for i,joint in enumerate(self.robot.actuated_joints):
            if joint.joint_type == 'continuous':
                self.continuous_joints.append(i)
            elif joint.joint_type in ['floating', 'planar']:
                raise NotImplementedError
            if joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else -np.Inf
                upper = joint.limit.upper if joint.limit.upper is not None else np.Inf
                pos_lim[i] = [lower, upper]
                vel_lim[i] = joint.limit.velocity
                eff_lim[i] = joint.limit.effort
        self.pos_lim = np.array(pos_lim).T
        self.vel_lim = np.array(vel_lim)
        self.eff_lim = np.array(eff_lim) # Unused for now
        # vars
        # State variables for the configuration of the URDF robot
        self.state = np.zeros(self.dof*2)
        self.init_state = np.zeros(self.dof*2)
        self.last_trajectory = self.qpos

        ### Simulation parameters for each step of this environment
        if step_type not in list(STEP_TYPE):
            raise ValueError
        self.step_type = step_type
        self._max_episode_steps = max_episode_steps
        self.t_step = t_step
        self.timestep_discretization = timestep_discretization
        self.integration_discretization = integration_discretization
        # vars
        self._elapsed_steps = 0
        self.terminal_observation = None

        ### Collision checking parameters
        self.check_collision = check_collision
        self.check_self_collision = check_self_collision
        self.verbose_self_collision = verbose_self_collision
        self.use_bb_collision = use_bb_collision
        self.collision_discretization = collision_discretization
        self.collision_use_visual = collision_use_visual
        # Add each of the bodies from the robot to a collision manager internal to this object.
        self.robot_collision_manager = trimesh.collision.CollisionManager()
        self.robot_collision_objs = set()
        for link in self.robot.links:
            if not self.collision_use_visual and link.collision_mesh is not None:
                mesh = link.collision_mesh.bounding_box if use_bb_collision else link.collision_mesh
                self.robot_collision_manager.add_object(link.name, mesh)
                self.robot_collision_objs.add(link.name)
            if self.collision_use_visual and len(link.visuals) > 0:
                # merge all visual meshes
                all_meshes = []
                for visual in link.visuals:
                    for mesh in visual.geometry.meshes:
                        pose = visual.origin
                        if visual.geometry.mesh is not None:
                            if visual.geometry.mesh.scale is not None:
                                S = np.eye(4)
                                S[:3,:3] = np.diag(visual.geometry.mesh.scale)
                                pose = pose.dot(S)
                        all_meshes.append(mesh.copy())
                        all_meshes[-1].apply_transform(pose)
                if len(all_meshes) == 0:
                    continue
                combined_mesh = all_meshes[0] + all_meshes[1:]
                mesh = combined_mesh.bounding_box if use_bb_collision else combined_mesh
                self.robot_collision_manager.add_object(link.name, mesh)
                self.robot_collision_objs.add(link.name)

        # vars
        self.collision_info = None

        ### Joint limit check parameters
        self.check_joint_limits = check_joint_limits
        self.enforce_joint_pos_limit = enforce_joint_pos_limit
        self.verbose_joint_limits = verbose_joint_limits
        self.joint_limit_info = None

        ### Visualization Parameters
        if renderer not in list(RENDERER):
            raise ValueError
        self.renderer = renderer
        self.render_frames = render_frames
        self.render_fps = render_fps
        self.render_size = render_size
        self.render_mesh = render_mesh
        self.renderer_kwargs = renderer_kwargs
        self.reopen_on_close = reopen_on_close
        # vars
        self.scene = None
        self.scene_map = None
        self.scene_viewer = None
        self._info = None
    
    def reset(self, qpos: np.ndarray = None, qvel: np.ndarray = 0, state: np.ndarray = None):
        # set or generate a state
        if state is not None:
            self.state = np.copy(state)
            self.qpos = self._wrap_cont_joints(self.qpos)
        else:
            if qpos is not None:
                self.qpos = self._wrap_cont_joints(qpos)
            else:
                # Generate a random position for each of the joints
                # Try 10 times until there is no self collision
                self.qpos = self._generate_free_configuration(n_tries=10)
            
            # Set the initial velocity to 0 or whatever is provided
            self.qvel = qvel
        self.init_state = np.copy(self.state)
        self.done = False
        self._elapsed_steps = 0
        self.last_trajectory = self.qpos
        self.terminal_observation = None
        self.collision_info = None
        self.joint_limit_info = None
        self._info = None

        return self.get_observations()
    
    def _generate_free_configuration(self, n_tries=10):
        # compute non-inf bounds
        pos_lim = np.copy(self.pos_lim)
        pos_lim[np.isneginf(pos_lim)] = -np.pi*3
        pos_lim[np.isposinf(pos_lim)] = np.pi*3

        # temporarily disable verbose self collisions
        original_verbosity = self.verbose_self_collision
        self.verbose_self_collision = False

        def pos_helper():
            new_pos = self.np_random.uniform(low=pos_lim[0], high=pos_lim[1])
            new_pos = self._wrap_cont_joints(new_pos)
            fk_dict = self.robot.link_fk(new_pos, use_names=True)
            collision, _ = self._self_collision_check(fk_dict)
            return collision, new_pos
        
        for _ in range(n_tries):
            collision, new_pos = pos_helper()
            if not collision:
                # restore verbosity
                self.verbose_self_collision = original_verbosity
                return new_pos
        raise RuntimeError

    def get_observations(self):
        observation = {
            'qpos': self.qpos,
            'qvel': self.qvel,
        }
        return copy.deepcopy(observation)
    
    def step(self, action):
        if self.step_type == STEP_TYPE.DIRECT:
            q = np.copy(action[0])
            qd = np.copy(action[1])
            if len(q) != self.timestep_discretization or len(qd) != self.timestep_discretization:
                raise RuntimeError
        elif self.step_type == STEP_TYPE.INTEGRATION:
            qdd = self._interpolate_q(action, self.integration_discretization)
            t = self.t_step / self.integration_discretization
            qd_delt = np.cumsum(qdd * t, axis=0)
            qd = self.qvel + qd_delt
            q_delt = np.cumsum((qd * t) + (0.5 * qdd * t * t), axis=0)
            q = self.qpos + q_delt
        # Perform joint limit check if desired before enforcing joint limits
        if self.check_joint_limits:
            self.joint_limit_info = self._joint_limit_check(q, qd)
        if self.enforce_joint_pos_limit:
            q, qd = self._enforce_joint_pos_limit(q, qd)
        # Perform collision checks
        if self.check_collision:
            self.collision_info = self._collision_check_internal(q)
        # Get the timestep discretization of each
        q = self._interpolate_q(q, self.timestep_discretization)
        qd = self._interpolate_q(qd, self.timestep_discretization)
        self.last_trajectory = q
        # wraparound the positions for continuous joints (do this after checks becuase checks do different discretizations)
        q = self._wrap_cont_joints(q)
        # store the final state
        self.qpos = q[-1]
        self.qvel = qd[-1]
        # Update extra variables and return
        self._elapsed_steps += 1
        observations = self.get_observations()
        self._info = self.get_info()
        reward = self.get_reward(action)
        self.done = self.done if self._elapsed_steps < self._max_episode_steps else True
        return observations, reward, self.done, self._info

    def get_reward(self, action):
        return 0
        
    def get_info(self):
        info = {
            'init_qpos': self.init_qpos,
            'init_qvel': self.init_qvel,
            'last_trajectory': self.last_trajectory
            }
        if self.check_collision:
            info['collision_info'] = self.collision_info
        if self.check_joint_limits:
            info['joint_limit_exceeded'] = self.joint_limit_info
        if self.done:
            if self.terminal_observation is None:
                self.terminal_observation = copy.deepcopy(self.get_observations())
            info['terminal_observation'] = self.terminal_observation
        return copy.deepcopy(info)
        
    def _collision_check_internal(self, q_step):
        out = {
            'in_collision': False,
        }
        # Interpolate the given for q for the number of steps we want
        q_check = self._interpolate_q(q_step, self.collision_discretization)
        batch_fk_dict = self.robot.link_fk_batch(cfgs=q_check, links=self.robot_collision_objs)
        merge = self.collision_check(batch_fk_dict)
        out.update(merge)
        # Check each of the configurations
        if self.check_self_collision:
            out['self_collision'] = False
            self_collision_pairs = {}
            for i in range(self.collision_discretization):
                # Comprehend a name: transform from the batch link fk dictionary
                fk_dict = {link.name: pose[i] for link, pose in batch_fk_dict.items()}
                collision, pairs = self._self_collision_check(fk_dict)
                if self.verbose_self_collision and collision:
                    collision_time = float(i)/self.collision_discretization * self.t_step
                    self_collision_pairs[collision_time] = pairs
                elif collision:
                    out['in_collision'] = True
                    out['self_collision'] = True
                    break
            if len(self_collision_pairs) > 0:
                out['in_collision'] = True
                out['self_collision'] = self_collision_pairs
                    
        # Return as a dict
        return out
    
    def collision_check(self, batch_fk_dict):
        return {}

    def render(self,
               render_frames = None,
               render_fps = None,
               render_size = None,
               render_mesh = None,
               renderer = None,
               renderer_kwargs = None):
        render_frames = render_frames if render_frames is not None else self.render_frames
        render_fps = render_fps if render_fps is not None else self.render_fps
        render_size = render_size if render_size is not None else self.render_size
        render_mesh = render_mesh if render_mesh is not None else self.render_mesh
        renderer = renderer if renderer is not None else self.renderer
        renderer_kwargs = renderer_kwargs if renderer_kwargs is not None else self.renderer_kwargs

        # Prep the q to render
        q_render = self._interpolate_q(self.last_trajectory, render_frames)
        q_render = self._wrap_cont_joints(q_render)
        fk = self.robot.visual_trimesh_fk_batch(q_render)

        # Generate the render
        if renderer == RENDERER.PYRENDER:
            import time
            if self.scene is None:
                self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
            for i in range(render_frames):
                proc_time_start = time.time()
                if not self.scene_viewer.is_active and self.reopen_on_close:
                    self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
                elif not self.scene_viewer.is_active:
                    return
                self.scene_viewer.render_lock.release()
                self.scene_viewer.render_lock.acquire()
                for mesh, node in self.scene_map.items():
                    pose = fk[mesh][i]
                    node.matrix = pose
                self.scene_viewer.render_lock.release()
                if render_fps is not None:
                    proc_time = time.time() - proc_time_start
                    pause_time = max(0, 1.0/render_fps - proc_time)
                    time.sleep(pause_time)
                self.scene_viewer.render_lock.acquire()
        elif renderer == RENDERER.PYRENDER_OFFSCREEN:
            if self.scene is None:
                self._create_pyrender_scene(render_mesh, render_size, render_fps, renderer_kwargs)
            color_frames = [None]*render_frames
            depth_frames = [None]*render_frames
            for i in range(render_frames):
                for mesh, node in self.scene_map.items():
                    pose = fk[mesh][i]
                    node.matrix = pose
                color_frames[i], depth_frames[i] = self.scene_viewer.render(self.scene)
            return color_frames, depth_frames
        
    def close(self):
        if self.renderer == RENDERER.PYRENDER:
            if self.scene_viewer is not None and self.scene_viewer.is_active:
                self.scene_viewer.close_external()
        elif self.renderer == RENDERER.PYRENDER_OFFSCREEN:
            if self.scene_viewer is not None:
                self.scene_viewer.delete()
        self.scene = None
        self.scene_map = None
        self.scene_viewer = None

    def _wrap_cont_joints(self, pos: np.ndarray) -> np.ndarray:
        pos = np.copy(pos)
        pos[..., self.continuous_joints] = (pos[..., self.continuous_joints] + np.pi) % (2 * np.pi) - np.pi
        return pos

    # Utility to interpolate the position of the joints to a desired number of steps
    def _interpolate_q(self, q: np.ndarray, out_steps):
        if len(q.shape) == 1:
            q = np.expand_dims(q,0)
        in_times = np.linspace(0, 0.5, num=len(q), endpoint=False)
        coll_times = np.linspace(0, 0.5, num=out_steps, endpoint=False)

        q_interpolated = [None]*self.dof
        for i in range(self.dof):
            q_interpolated[i] = np.interp(coll_times, in_times, q[:,i])
        q_interpolated = np.array(q_interpolated, order='F').T # Use fortran order so it's C order when transposed.
        return q_interpolated

    def _self_collision_check(self, fk_dict):
        # Use the collision manager to check each of these positions
        for name, transform in fk_dict.items():
            if name in self.robot_collision_objs:
                self.robot_collision_manager.set_transform(name, transform)
        collision, name_pairs = self.robot_collision_manager.in_collision_internal(return_names=True)
        # if we have collision, make sure it isn't pairwise
        if collision:
            # check all name pairs
            true_names = set()
            for n1, n2 in name_pairs:
                link1 = self.robot.link_map[n1]
                link2 = self.robot.link_map[n2]
                # If the links aren't neighbors in the underlying robot graph, there's
                # a true collision
                if link2 not in self.robot_graph[link1]:
                    if self.verbose_self_collision:
                        true_names.add((n1, n2))
                    else:
                        return True, set()
            return len(true_names) > 0, true_names
        return False, set()

    def _joint_limit_check(self, q_step, qd_step):
        pos_exceeded = np.logical_or(q_step < self.pos_lim[0], q_step > self.pos_lim[1])
        vel_exceeded = np.abs(qd_step) > self.vel_lim
        exceeded = np.any(pos_exceeded) or np.any(vel_exceeded)
        out = {'exceeded': exceeded}
        if exceeded and self.verbose_joint_limits:
            def nonzero_to_odict(times, joints):
                n_times = len(q_step)
                times = times.astype(float) / n_times * self.t_step
                exceeded_dict = OrderedDict()
                for t, j in zip(times, joints):
                    joint_list = exceeded_dict.get(t,[])
                    joint_list.append(j)
                    exceeded_dict[t] = joint_list
                return exceeded_dict
            times, joints = np.nonzero(pos_exceeded)
            out['pos_exceeded'] = nonzero_to_odict(times, joints)
            times, joints = np.nonzero(vel_exceeded)
            out['vel_exceeded'] = nonzero_to_odict(times, joints)
        return out
    
    def _enforce_joint_pos_limit(self, q_step, qd_step):
        q_step = np.copy(q_step)
        qd_step = np.copy(qd_step)
        # Lazy enforce lower
        pos_exceeded_lower = q_step < self.pos_lim[0]
        q_step[pos_exceeded_lower] = np.broadcast_to(self.pos_lim[0], q_step.shape)[pos_exceeded_lower]
        qd_step[pos_exceeded_lower] = 0
        # Lazy enforce upper
        pos_exceeded_upper = q_step > self.pos_lim[1]
        q_step[pos_exceeded_upper] = np.broadcast_to(self.pos_lim[1], q_step.shape)[pos_exceeded_upper]
        qd_step[pos_exceeded_upper] = 0
        return q_step, qd_step

    def _create_pyrender_scene(self, render_mesh, render_size, render_fps, renderer_kwargs):
        import pyrender
        self.scene = pyrender.Scene()
        fk = self.robot.visual_trimesh_fk()
        self.scene_map = {}
        for tm, pose in fk.items():
            if render_mesh:
                mesh = pyrender.Mesh.from_trimesh(tm, smooth=False)
            else:
                mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False)
            node = self.scene.add(mesh, pose=pose)
            self.scene_map[tm] = node
        if self.renderer == RENDERER.PYRENDER:
            # Default viewport arguments
            kwargs = {
                'viewport_size': render_size,
                'run_in_thread': True,
                'use_raymond_lighting': True,
                'refresh_rate': render_fps
            }
            kwargs.update(renderer_kwargs)
            self.scene_viewer = pyrender.Viewer(
                                    self.scene,
                                    **kwargs)
            self.scene_viewer.render_lock.acquire()
        else:
            w, h = render_size
            ### compute a camera (from the pyrender viewer code)
            from pyrender.constants import DEFAULT_Z_FAR, DEFAULT_Z_NEAR, DEFAULT_SCENE_SCALE
            ## camera pose
            centroid = self.scene.centroid
            if renderer_kwargs.get('view_center') is not None:
                centroid = renderer_kwargs['view_center']
            scale = self.scene.scale    
            if scale == 0.0:
                scale = DEFAULT_SCENE_SCALE
            s2 = 1.0 / np.sqrt(2.0)
            cp = np.eye(4)
            cp[:3,:3] = np.array([
                [0.0, -s2, s2],
                [1.0, 0.0, 0.0],
                [0.0, s2, s2]
            ])
            hfov = np.pi / 6.0
            dist = scale / (2.0 * np.tan(hfov))
            cp[:3,3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
            ## camera perspective
            zfar = max(self.scene.scale * 10.0, DEFAULT_Z_FAR)
            if self.scene.scale == 0:
                znear = DEFAULT_Z_NEAR
            else:
                znear = min(self.scene.scale / 10.0, DEFAULT_Z_NEAR)
            ## camera node
            camera = pyrender.camera.PerspectiveCamera(yfov=np.pi / 3.0, znear=znear, zfar=zfar)
            camera_node = pyrender.Node(matrix=cp, camera=camera)
            self.scene.add_node(camera_node)
            self.scene.main_camera_node = camera_node
            ## raymond lights
            thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
            phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
            for phi, theta in zip(phis, thetas):
                xp = np.sin(theta) * np.cos(phi)
                yp = np.sin(theta) * np.sin(phi)
                zp = np.cos(theta)

                z = np.array([xp, yp, zp])
                z = z / np.linalg.norm(z)
                x = np.array([-z[1], z[0], 0.0])
                if np.linalg.norm(x) == 0:
                    x = np.array([1.0, 0.0, 0.0])
                x = x / np.linalg.norm(x)
                y = np.cross(z, x)

                matrix = np.eye(4)
                matrix[:3,:3] = np.c_[x,y,z]
                self.scene.add_node(pyrender.Node(
                    light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                    matrix=matrix
                ))
            ### Camera computed

            kwargs = OrderedDict()
            kwargs['viewport_width'] = w
            kwargs['viewport_height'] = h
            kwargs.update(renderer_kwargs)
            self.scene_viewer = pyrender.OffscreenRenderer(**kwargs)

    @property
    def action_spec(self):
        pass
    @property
    def action_dim(self):
        pass
    @property 
    def action_space(self):
        pass 
    @property 
    def observation_space(self):
        pass 
    @property 
    def obs_dim(self):
        pass

    @property
    def info(self):
        return copy.deepcopy(self._info)

    # State related properties and setters
    @property
    def qpos(self) -> np.ndarray:
        return self.state[:self.dof]
    
    @qpos.setter
    def qpos(self, val: np.ndarray):
        self.state[:self.dof] = val

    @property
    def qvel(self) -> np.ndarray:
        return self.state[self.dof:]
    
    @qvel.setter
    def qvel(self, val: np.ndarray):
        self.state[self.dof:] = val
    
    @property
    def init_qpos(self) -> np.ndarray:
        return self.init_state[:self.dof]

    @property
    def init_qvel(self) -> np.ndarray:
        return self.init_state[self.dof:]


if __name__ == '__main__':

    # Load robot
    import os
    basedirname = os.path.dirname(zp.__file__)

    print('Loading Robot')
    # This is hardcoded for now
    import zonopy.robots2.robot as robots2
    robots2.DEBUG_VIZ = False
    rob = robots2.ArmRobot(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'))

    test = KinematicUrdfBase(rob.robot, step_type='integration', check_joint_limits=True, check_self_collision=True, use_bb_collision=True, render_mesh=True, reopen_on_close=False)
    test.render()
    test.reset()
    # plt.figure()
    # plt.ion()
    for i in range(302):
        a = test.step(np.random.random(7)-0.5)
        test.render()
        # im, depth = test.render()
        # for i in range(0,len(im),10):
        #     plt.imshow(im[i])
        #     plt.draw()
        #     plt.pause(0.01)
        # test.render(render_fps=5)
        print(a)

    print("hi")

    # env = Locked_Arm_3D(n_obs=3,T_len=50,interpolate=True,locked_idx=[1,2],locked_qpos = [0,0])
    # for _ in range(3):
    #     for _ in range(10):
    #         env.step(torch.rand(env.dof))
    #         env.render()
    #         #env.reset()
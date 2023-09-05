# TODO DOCUMNET & CLEAN

from zonopy.environments.urdf_base import KinematicUrdfBase, STEP_TYPE, RENDERER
import numpy as np
import trimesh

class KinematicUrdfWithObstacles(KinematicUrdfBase):
    def __init__(self,
                 n_obs = 1, # number of obstacles
                 obs_size_max = [0.1,0.1,0.1], # maximum size of randomized obstacles in xyz
                 obs_size_min = [0.1,0.1,0.1], # minimum size of randomized obstacle in xyz
                 obs_gen_buffer = 1e-4, # buffer around obstacles generated
                 goal_threshold = 0.1, # goal threshold
                 **kwargs):
        super().__init__(**kwargs)

        self.n_obs = n_obs
        self.obs_size_range = np.array([obs_size_min, obs_size_max])
        if len(self.obs_size_range) != 3:
            self.obs_size_range = np.broadcast_to(np.expand_dims(self.obs_size_range,1), (2, self.n_obs, 3))
        self.obs_gen_buffer = obs_gen_buffer
        self.goal_threshold = goal_threshold
        self.qgoal = np.zeros(self.dof)

        self.obstacle_collision_manager = None
        self.obs_list = []
        self.obs_size = None
        self.obs_pos = None

    def reset(self,
              qgoal: np.ndarray = None,
              obs_pos: np.ndarray = None,
              obs_size: np.ndarray = None,
              **kwargs):
        # Setup the initial environment
        super().reset(**kwargs)

        # Generate the goal position
        if qgoal is not None:
            self.qgoal = self._wrap_cont_joints(qgoal)
        else:
            # Generate a random position for each of the joints
            # Try 10 times until there is no self collision
            self.qgoal = self._generate_free_configuration(n_tries=10)

        # generate the obstacle sizes we want
        if obs_size is None:
            obs_size = self.np_random.uniform(low=self.obs_size_range[0], high=self.obs_size_range[1])
        obs_size = np.asarray(obs_size)

        # if no position is provided, generate
        if obs_pos is None:
            # create temporary collision managers for start and end
            start_collision_manager = trimesh.collision.CollisionManager()
            goal_collision_manager = trimesh.collision.CollisionManager()
            start_fk = self.robot.link_fk(self.qpos)
            goal_fk = self.robot.link_fk(self.qgoal)
            for (link, start_tf), goal_tf in zip(start_fk.items(), goal_fk.values()):
                if link.collision_mesh is not None:
                    mesh = link.collision_mesh
                    start_collision_manager.add_object(link.name, mesh, transform=start_tf)
                    goal_collision_manager.add_object(link.name, mesh, transform=goal_tf)
                    
            # For each of the obstacles, determine a position.
            # Generate arm configurations and use the end effector location as a candidate position
            # compute non-inf bounds
            pos_lim = np.copy(self.pos_lim)
            pos_lim[np.isneginf(pos_lim)] = -np.pi*3
            pos_lim[np.isposinf(pos_lim)] = np.pi*3
            def pos_helper(mesh):
                new_pos = self.np_random.uniform(low=pos_lim[0], high=pos_lim[1])
                new_pos = self._wrap_cont_joints(new_pos)
                fk_dict = self.robot.link_fk(new_pos, links=self.robot.end_links)
                cand_pos = np.array(list(fk_dict.values()))
                cand_pos[...,0:3,0:3] = np.eye(3)
                for pose in cand_pos:
                    coll_start = start_collision_manager.in_collision_single(mesh, pose)
                    coll_end = start_collision_manager.in_collision_single(mesh, pose)
                    if not (coll_start or coll_end):
                        return pose[0:3,3]
                return None
            
            # generate obstacle
            obs_pos_list = []
            for size in obs_size:
                # Try 10 times per obstacle
                pos = None
                for _ in range(10):
                    cand_mesh = trimesh.primitives.Box(extents=size + self.obs_gen_buffer * 2)
                    pos = pos_helper(cand_mesh)
                    if pos is not None:
                        break
                if pos is None:
                    raise RuntimeError
                obs_pos_list.append(pos)
            obs_pos = np.array(obs_pos_list)
        obs_pos = np.asarray(obs_pos)
        
        # add the obstacles to the obstacle collision manager
        self.obs_pos = obs_pos
        self.obs_size = obs_size
        self.obstacle_collision_manager = trimesh.collision.CollisionManager()
        self.obs_meshes = {}
        for i, (pos, size) in enumerate(zip(obs_pos, obs_size)):
            obs = trimesh.primitives.Box(extents=size)
            pose = np.eye(4)
            pose[0:3,3] = pos
            obs_name = f'obs{i}'
            self.obstacle_collision_manager.add_object(obs_name, obs, transform=pose)
            self.obs_meshes[obs_name] = (obs, pose)
        
        # reset the pyrender becuase I don't feel like tracking obstacles
        self.close()
        
        # return observations
        return self.get_observations()
    
    def get_reward(self, action):
        # Get the position and goal then calculate distance to goal
        collision = self.info['collision_info']['in_collision']
        dist = np.linalg.norm(self._wrap_cont_joints(self.qpos - self.qgoal))
        success = (dist < self.goal_threshold) and (not collision) and (np.linalg.norm(self.qvel) < 0)

        self.done = self.done or success or collision

        if success:
            return 1
        else:
            return 0
    
    def get_observations(self):
        observations = super().get_observations()
        observations['qgoal'] = self.qgoal
        observations['obstacle_pos'] = self.obs_pos
        observations['obstacle_size'] = self.obs_size
        return observations
    
    def _create_pyrender_scene(self, *args, **kwargs):
        import pyrender
        super()._create_pyrender_scene(*args, **kwargs)
        # add the obstacles
        obs_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.4,
            alphaMode='BLEND',
            baseColorFactor=(1.0, 0.0, 0.0, 0.3),
        )
        for mesh, pose in self.obs_meshes.values():
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, wireframe=True, material=obs_mat)
            self.scene.add(pyrender_mesh, pose=pose)
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=obs_mat)
            self.scene.add(pyrender_mesh, pose=pose)
        # Add the goal
        goal_mat = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='BLEND',
            baseColorFactor=(0.0, 1.0, 0.0, 0.3),
        )
        fk = self.robot.visual_trimesh_fk(self.qgoal)
        for tm, pose in fk.items():
            mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False, wireframe=True, material=goal_mat)
            self.scene.add(mesh, pose=pose)
            mesh = pyrender.Mesh.from_trimesh(tm.bounding_box, smooth=False, material=goal_mat)
            self.scene.add(mesh, pose=pose)

    def _collision_check_obstacle(self, fk_dict):
        # Use the collision manager to check each of these positions
        for name, transform in fk_dict.items():
            if name in self.robot_collision_objs:
                self.robot_collision_manager.set_transform(name, transform)
        collision, name_pairs = self.robot_collision_manager.in_collision_other(self.obstacle_collision_manager, return_names=True)
        return collision, name_pairs

    def collision_check(self, batch_fk_dict):
        out = {}
        out['obstacle_collision'] = False
        collision_pairs = {}
        for i in range(self.collision_discretization):
            # Comprehend a name: transform from the batch link fk dictionary
            fk_dict = {link.name: pose[i] for link, pose in batch_fk_dict.items()}
            collision, pairs = self._collision_check_obstacle(fk_dict)
            if self.verbose_self_collision and collision:
                collision_time = float(i)/self.collision_discretization * self.t_step
                collision_pairs[collision_time] = pairs
            elif collision:
                out['in_collision'] = True
                out['obstacle_collision'] = True
                break
        if len(collision_pairs) > 0:
            out['in_collision'] = True
            out['obstacle_collision'] = collision_pairs
        return out

if __name__ == '__main__':

    # Load robot
    import os
    import zonopy as zp
    basedirname = os.path.dirname(zp.__file__)

    print('Loading Robot')
    # This is hardcoded for now
    import zonopy.robots2.robot as robots2
    robots2.DEBUG_VIZ = False
    rob = robots2.ZonoArmRobot(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'))
    # rob = robots2.ArmRobot('/home/adamli/rtd-workspace/urdfs/panda_arm/panda_arm_proc.urdf')

    test = KinematicUrdfWithObstacles(robot = rob.robot, step_type='integration', check_joint_limits=True, check_self_collision=True, use_bb_collision=True, render_mesh=True, reopen_on_close=False, n_obs=5, render_fps=30, render_frames=10)
    # test.render()
    test.reset()
    # test.reset(
    #     qpos=np.array([ 3.1098, -0.9964, -0.2729, -2.3615,  0.2724, -1.6465, -0.5739]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([-1.9472,  1.4003, -1.3683, -1.1298,  0.7062, -1.0147, -1.1896]),
    #     obs_pos=[
    #         np.array([ 0.3925, -0.7788,  0.2958]),
    #         np.array([0.3550, 0.3895, 0.3000]),
    #         np.array([-0.0475, -0.1682, -0.7190]),
    #         np.array([0.3896, 0.5005, 0.7413]),
    #         np.array([0.4406, 0.1859, 0.1840]),
    #         np.array([ 0.1462, -0.6461,  0.7416]),
    #         np.array([-0.4969, -0.5828,  0.1111]),
    #         np.array([-0.0275,  0.1137,  0.6985]),
    #         np.array([ 0.4139, -0.1155,  0.7733]),
    #         np.array([ 0.5243, -0.7838,  0.4781])
    #         ])
    # test.reset(
    #     qpos=np.array([-1.3030, -1.9067,  2.0375, -1.5399, -1.4449,  1.5094,  1.9071]),
    #     qvel=np.array([0,0,0,0,0,0,0.]),
    #     qgoal = np.array([ 0.7234,  1.6843,  2.5300, -1.0317, -3.1223,  1.2235,  1.3428]),
    #     obs_pos=[
    #         np.array([0.65,-0.46,0.33]),
    #         np.array([0.5,-0.43,0.3]),
    #         np.array([0.47,-0.45,0.15]),
    #         np.array([-0.3,0.2,0.23]),
    #         np.array([0.3,0.2,0.31])
    #         ])
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
        # print(a)

    print("hi")

    # env = Locked_Arm_3D(n_obs=3,T_len=50,interpolate=True,locked_idx=[1,2],locked_qpos = [0,0])
    # for _ in range(3):
    #     for _ in range(10):
    #         env.step(torch.rand(env.dof))
    #         env.render()
    #         #env.reset()
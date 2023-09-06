from urchin import URDF, xyz_rpy_to_matrix
import numpy as np
import trimesh
from typing import Union
import torch
import zonopy as zp

# NOTE: WIP

# Function to create a 3D basis for any defining normal vector (to an arbitrary hyperplane)
# Returns the basis as column vectors
def normal_vec_to_basis(norm_vec):
    # first normalize the vector
    norm_vec = np.array(norm_vec, dtype=float).squeeze()
    norm_vec = norm_vec / np.linalg.norm(norm_vec)

    # Helper function for simple basis with unitary elements
    def simple_basis(order):
        ret = np.eye(3)
        idx = (np.arange(3) + order) % 3
        return ret[:,idx]

    # Try to project [1, 0, 0]
    if (proj := np.dot([1.0, 0, 0], norm_vec)):
        # Use this vector to create an orthogonal component
        rej = np.array([1.0, 0, 0]) - (norm_vec * proj)
        # Case for normal vector of [1, 0, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(1)
    # If not, try to project [0, 1, 0] and do the same
    elif (proj := np.dot([0, 1.0, 0], norm_vec)):
        rej = np.array([0, 1.0, 0]) - (norm_vec * proj)
        # Case for normal vector of [0, 1, 0]
        if np.linalg.norm(rej) == 0:
            return simple_basis(2)
    else:
        # Otherwise, we are dealing with normal vector of [0, 0, 1],
        # so just create the identity as the basis
        return simple_basis(3)
    
    # Find a third orthogonal vector
    cross = np.cross(rej, norm_vec)
    # Just for simplicity, we treat the cross as x, the rej as y, and the vec as z
    # in order to keep a properly left-handed basis
    cross = cross / np.linalg.norm(cross)
    rej = rej / np.linalg.norm(rej)
    return np.column_stack((cross, rej, norm_vec))
    

# Some variables for computing the joint radius
JOINT_MOTION_UNION_ANGLES = np.array([30, 60, 90], dtype=float) * np.pi / 180.0
JOINT_INTERSECT_COUNT = 3
INLINE_RATIO_CUTOFF = 1.5
DEBUG_VIZ = True

# This assumes simple chain kinematics, no branching has been considered
# See section on joint bounds
def load_robot(filepath):
    robot = URDF.load(filepath)

    # # Preprocess link parent joints and add that to a map
    robot.link_parent_joint = {robot.base_link.name: None}
    for joint in robot.joints:
        # Joint must have parent or child
        robot.link_parent_joint[joint.child] = joint
        # Also make an origin tensor
        joint.origin_tensor = torch.as_tensor(joint.origin, dtype=torch.get_default_dtype())


    ##### THE ORIGIN DENOTES THE JOINT TO JOINT AXIS!
    ##### In order to ensure that the following joint's radius works for the prior link.
    # we should instead make it a 2D projection of the link along the origin axis.
    # This can then be used to do that joint occupancy, with ea


    # Get the axis-aligned bounding boxes for all links
    # This assumes the home position is aligned.
    # This assumes simple chain kinematics, no branching has been considered
    # Iterate all joints
    # This only returns the interval, but could be modified to get a ball or similar.
    for joint in robot.joints:
        # Get the trimesh and transform for the collision for multiple configurations
        if joint.joint_type in ['prismatic', 'floating', 'planar']:
            raise NotImplementedError
        # Get the links we care about too
        parent_link = robot.link_map[joint.parent]
        child_link = robot.link_map[joint.child]
        
        # FLag if we are going to treat it as a fixed joint
        # For the fixed joint, we want intersections of three orthogonal rotations.
        # So to acheive that, we just add one more orthogonal rotation.
        treat_fixed = None

        # Do an initial union and bounding box.
        res = robot.collision_trimesh_fk(links=[parent_link, child_link])
        # if we only have one mesh, rotate 180 and then treat it as fixed
        if len(res) != 2:
            basis = normal_vec_to_basis(joint.axis)
            rotmat = np.hstack(([0,0,0],basis[:,0])) * np.pi
            rotmat = xyz_rpy_to_matrix(rotmat)
            first_el = list(res.items())[0]
            res[first_el[0].copy().apply_transform(rotmat)] = first_el[1]
            child_volume = first_el[0].bounding_box.volume
            treat_fixed = True
        else:
            child_volume = child_link.collision_mesh.bounding_box.volume
        res = [mesh.copy().apply_transform(transform) for mesh, transform in res.items()]
        combined_mesh = trimesh.util.concatenate(res[0].bounding_box, res[1].bounding_box)
        combined_mesh_debug = trimesh.util.concatenate(res[0], res[1]) if DEBUG_VIZ else None
        base_volume = combined_mesh.bounding_box.volume
        child_volume_ratio = child_volume / base_volume

        # Move and union the bounding boxes for links in the joint if it's moveable
        if joint.joint_type in ['revolute', 'continuous'] and not treat_fixed:
            for ang in JOINT_MOTION_UNION_ANGLES:
                res = robot.collision_trimesh_fk(cfg={joint.name: ang}, links=[child_link])
                res = [mesh.copy().apply_transform(transform) for mesh, transform in res.items()]
                combined_mesh = trimesh.util.concatenate(combined_mesh, res[0].bounding_box)
            # Check if it was mostly inline by taking the ratio of added volume to that of a child link
            added_ratio = combined_mesh.bounding_box.volume / base_volume - 1
            ratio = added_ratio / child_volume_ratio
            treat_fixed = (ratio < INLINE_RATIO_CUTOFF)
        
        # Get the child base transform, and invert to get the joint at the origin
        res = robot.link_fk(links=[child_link])
        base_transform = list(res.values())[0]

        combined_mesh.apply_transform(np.linalg.pinv(base_transform))
        if DEBUG_VIZ:
            combined_mesh_debug.apply_transform(np.linalg.pinv(base_transform))

        ###
        # This section didn't work right, and was unreliable.
        ###
        # Do the rotations (create a transform for 90 degree rotations)
        # trimmed_mesh = combined_mesh.copy().bounding_box
        # rot_transform = np.hstack(([0,0,0], joint.axis)) * np.pi * 0.5
        # rot_transform = xyz_rpy_to_matrix(rot_transform)
        # for _ in range(JOINT_INTERSECT_COUNT):
        #     rot_mesh = trimmed_mesh.copy().apply_transform(rot_transform)
        #     trimmed_mesh = trimesh.boolean.boolean_automatic([trimmed_mesh, rot_mesh], 'intersection').bounding_box

        # # Do additional orthogonal rotations of the trimed mesh if it is fixed, or we are treating it as fixed
        # if joint.joint_type == 'fixed' or treat_fixed:
        #     basis = normal_vec_to_basis(joint.axis)
        #     xyzrpy = np.hstack((np.zeros((2, 3)), basis[:,:2].T)) * np.pi * 0.5
        #     for transform in xyzrpy:
        #         rot_transform = xyz_rpy_to_matrix(transform)
        #         # Adding bounding_box here addresses a nefarious bug where it doesn't actually create
        #         # valid geometry for blender to work with.
        #         rot_mesh = trimmed_mesh.copy().apply_transform(rot_transform).bounding_box
        #         trimmed_mesh = trimesh.boolean.boolean_automatic([trimmed_mesh, rot_mesh], 'intersection').bounding_box

        # Instead, since we're now centered on the joint, just obtain the min and max of the
        # absolute values of each vector element
        aabb_verts = combined_mesh.bounding_box.vertices
        extents = np.abs(aabb_verts)
        extents = np.column_stack([np.amin(extents, axis=0), np.amax(extents, axis=0)])

        # For the dimensions coplaner to the plane described by the axis normal, we take the min of the extends
        basis = normal_vec_to_basis(joint.axis)
        # Right now I only consider axis aligned situations, so throw for other cases.
        if not np.all(np.any(basis==1, axis=0)):
            raise NotImplementedError
        # Isolate the off axis components & take the combined min of the absolute extents
        _, basis_order = np.nonzero(basis.T)
        joint_bb = np.zeros((3,2))
        joint_bb[basis_order[:2],1] = np.min(extents[basis_order[:2],0])
        joint_bb[basis_order[:2],0] = -joint_bb[basis_order[:2],1]
        # If it's something we treat as fixed, compare to the min of the on-axis components
        # and make that it
        if joint.joint_type == 'fixed' or treat_fixed:
            on_axis = np.min(extents[:2,0])
            joint_bb[basis_order[2],:] = [-on_axis, on_axis]
        else:
            # For the on-axis component, take the aabb_bounds
            aabb_bounds = np.column_stack([np.amin(aabb_verts, axis=0), np.amax(aabb_verts, axis=0)])
            joint_bb[basis_order[2],:] = aabb_bounds[basis_order[2],:]

        if DEBUG_VIZ:
            print("Treat Fixed: ", treat_fixed)
            mesh = combined_mesh_debug
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
            mesh = trimesh.primitives.Box(bounds=joint_bb.T)
            ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, alpha=0.2)
            interval_size = np.max(np.abs(joint_bb), axis=1)
            mesh = trimesh.primitives.Box(bounds=[-interval_size, interval_size])
            ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, alpha=0.2)
            plt.show()

        # This doesn't create the right bounds for the last one, but that's okay for now.
        joint.radius = joint_bb[:,1]
        joint.aabb = joint_bb

        # Store zonotopes for the outer_bb and the aabb
        outer_rad = np.max(joint.radius)
        joint.outer_pz = zp.polyZonotope(torch.vstack([torch.zeros(3), torch.eye(3)*outer_rad]))
        center = np.sum(joint.aabb,axis=1) / 2
        gens = np.diag(joint.aabb[:,1] - center)
        joint.bounding_pz = zp.polyZonotope(torch.as_tensor(np.vstack([center,gens]), dtype=torch.get_default_dtype()))
        # joint.bounding_pz = 
        # test = a.collision_trimesh_fk(cfg={'joint_4':3.14/2.0},links=[link])

    # Create pz bounding boxes for each link
    for link in robot.links:
        try:
            trimesh_bb = link.collision_mesh.bounding_box
            bounds = trimesh_bb.vertices
            bounds = np.column_stack([np.amin(bounds, axis=0), np.amax(bounds, axis=0)])
        except AttributeError:
            # If there's no collision mesh, then make it just a 5cm square cube.
            bounds = np.ones(3)*0.05
            bounds = np.column_stack([-bounds, bounds])
        # Create the zonotope
        center = np.sum(bounds,axis=1) / 2
        gens = np.diag(bounds[:,1] - center)
        link_pz = zp.polyZonotope(torch.as_tensor(np.vstack([center,gens]), dtype=torch.get_default_dtype()))
        link.bounding_pz = link_pz

    return robot

import copy

class ZonoArmRobot:
    __slots__ = [
        'urdf',
        'dof',
        'np',
        'tensor',
        'device',
        'dtype',
        'joint_axis',
        'joint_origins',
        'pos_lim',
        'vel_lim',
        'eff_lim',
        'continuous_joints',
        'pos_lim_mask',
        'link_parent_joint',
        'joint_data',
        'link_data',
        '__joint_axis',
        '__joint_origins',
        '__pos_lim',
        '__vel_lim',
        '__eff_lim',
        '__continuous_joints',
        '__pos_lim_mask',
        '__joint_axis_np',
        '__joint_origins_np',
        '__pos_lim_np',
        '__vel_lim_np',
        '__eff_lim_np',
        '__continuous_joints_np',
        '__pos_lim_mask_np',
        '__basetype'
        ]
    
    def __init__(self):
        pass

    @staticmethod
    def load(robot: Union[URDF, str], device=None, dtype=None, itype=None, create_joint_occupancy=False):
        # Resolve the dtype and device to use
        temp_device = torch.empty(0, device=device)
        device = temp_device.device
        temp_dtype = torch.empty(0, device='cpu', dtype=dtype)
        dtype = temp_dtype.dtype
        np_dtype = temp_dtype.numpy().dtype
        if itype is not None:
            temp_itype = torch.empty(0, dtype=dtype, device='cpu')
        else:
            temp_itype = torch.tensor([0], device='cpu')
        itype = temp_itype.dtype
        np_itype = temp_itype.numpy().dtype
        
        # Prepare the robot
        if type(robot) == str:
            robot = URDF.load(robot)
        constructed = ZonoArmRobot()
        constructed.urdf = robot
        constructed.dof = len(robot.actuated_joints)
        constructed._setup_robot_actuated_joint_data(robot, np_dtype, np_itype)
        constructed._setup_robot_joint_data(robot, dtype, create_joint_occupancy)
        constructed._setup_robot_link_data(robot, dtype)
        constructed.__basetype = temp_dtype
        constructed.np = constructed.numpy()
        constructed.tensor = constructed.to(device='cpu')
        constructed = constructed.to(device)
        return constructed

    def _setup_robot_actuated_joint_data(self, robot, np_dtype, np_itype):
        continuous_joints = []
        pos_lim = [[-np.Inf, np.Inf]]*self.dof
        vel_lim = [np.Inf]*self.dof
        eff_lim = [np.Inf]*self.dof
        joint_axis = []
        joint_origins = []
        for i,joint in enumerate(robot.actuated_joints):
            if joint.joint_type == 'continuous':
                continuous_joints.append(i)
            elif joint.joint_type in ['floating', 'planar']:
                raise NotImplementedError
            if joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else -np.Inf
                upper = joint.limit.upper if joint.limit.upper is not None else np.Inf
                pos_lim[i] = [lower, upper]
                vel_lim[i] = joint.limit.velocity
                eff_lim[i] = joint.limit.effort
            joint_axis.append(joint.axis)
            joint_origins.append(joint.origin)
        
        # initial numpy conversion
        self.__joint_axis_np = np.array(joint_axis, dtype=np_dtype)
        self.__joint_origins_np = np.array(joint_origins, dtype=np_dtype)
        self.__pos_lim_np = np.array(pos_lim, dtype=np_dtype).T
        self.__vel_lim_np = np.array(vel_lim, dtype=np_dtype)
        self.__eff_lim_np = np.array(eff_lim, dtype=np_dtype) # Unused for now
        self.__continuous_joints_np = np.array(continuous_joints, dtype=np_itype)
        self.__pos_lim_mask_np = np.isfinite(self.__pos_lim_np).any(axis=0)

        # Create tensor references to that memory
        self.__joint_axis = torch.from_numpy(self.__joint_axis_np)
        self.__joint_origins = torch.from_numpy(self.__joint_origins_np)
        self.__pos_lim = torch.from_numpy(self.__pos_lim_np)
        self.__vel_lim = torch.from_numpy(self.__vel_lim_np)
        self.__eff_lim = torch.from_numpy(self.__eff_lim_np)
        self.__continuous_joints = torch.from_numpy(self.__continuous_joints_np)
        self.__pos_lim_mask = torch.from_numpy(self.__pos_lim_mask_np)
    
    def _setup_robot_joint_data(self, robot, dtype, create_joint_occupancy=False):
        # Map from each link to the parent joint
        self.link_parent_joint = {robot.base_link.name: None}
        self.joint_data = {}
        
        for joint in robot.joints:
            # Joint must have parent or child
            self.link_parent_joint[joint.child] = joint
            
            # Get the index
            try:
                idx = robot.actuated_joints.index(joint)
            except ValueError:
                idx = None
        
            # create the object
            single_joint_data = ZonoArmRobotJoint(idx, create_joint_occupancy)
            # Create a joint occupancy by rotating stuff
            # Only do this if we want it
            if create_joint_occupancy:
                radius_np, aabb_np = _generate_joint_occupancy(robot, joint)
                single_joint_data._radius = torch.as_tensor(radius_np, dtype=dtype, device='cpu')
                single_joint_data._aabb = torch.as_tensor(aabb_np, dtype=dtype, device='cpu')
                single_joint_data._radius_np = single_joint_data._radius.numpy()
                single_joint_data._aabb_np = single_joint_data._aabb.numpy()
                outer_rad = np.max(radius_np)
                single_joint_data._outer_pz = zp.polyZonotope(torch.vstack([torch.zeros(3), torch.eye(3)*outer_rad]), dtype=dtype, device='cpu') # TODO consider bringing back itype?
                center = np.sum(aabb_np, axis=1) / 2
                gens = np.diag(aabb_np[:,1] - center)
                single_joint_data._bounding_pz = zp.polyZonotope(np.vstack([center,gens]), dtype=dtype, device='cpu')

            # save origin and axis as tensors
            single_joint_data._axis = torch.as_tensor(joint.axis, dtype=dtype, device='cpu')
            single_joint_data._origin = torch.as_tensor(joint.origin, dtype=dtype, device='cpu')
            single_joint_data._axis_np = single_joint_data._axis.numpy()
            single_joint_data._origin_np = single_joint_data._origin.numpy()
            
            single_joint_data.np = single_joint_data.numpy()
            single_joint_data.tensor = single_joint_data.to(device='cpu')

            self.joint_data[joint] = single_joint_data

        
    def _setup_robot_link_data(self, robot, dtype):
        # Create pz bounding boxes for each link
        self.link_data = {}
        for link in robot.links:
            try:
                trimesh_bb = link.collision_mesh.bounding_box
                bounds = trimesh_bb.vertices
                bounds = np.column_stack([np.amin(bounds, axis=0), np.amax(bounds, axis=0)])
            except AttributeError:
                # If there's no collision mesh, then make it just a 5cm square cube.
                bounds = np.ones(3)*0.05
                bounds = np.column_stack([-bounds, bounds])
            # Create the zonotope
            center = np.sum(bounds,axis=1) / 2
            gens = np.diag(bounds[:,1] - center)
            Z = np.vstack([center,gens])

            single_link_data = ZonoArmRobotLink()
            single_link_data._bounding_pz = zp.polyZonotope(Z, dtype=dtype, device='cpu')

            single_link_data._mass = torch.as_tensor(link.inertial.mass, dtype=dtype, device='cpu')
            single_link_data._center_mass = torch.as_tensor(link.inertial.origin[0:3,3], dtype=dtype, device='cpu')
            single_link_data._inertia = torch.as_tensor(link.inertial.inertia, dtype=dtype, device='cpu')
            single_link_data._mass_np = single_link_data._mass.numpy()
            single_link_data._center_mass_np = single_link_data._center_mass.numpy()
            single_link_data._inertia_np = single_link_data._inertia.numpy()
            
            single_link_data.np = single_link_data.numpy()
            single_link_data.tensor = single_link_data.to(device='cpu')

            self.link_data[link] = single_link_data

    def numpy(self):
        # create a shallow copy of self
        ret = copy.copy(self)
        # update references to all the properties we care about
        ret.joint_axis = self.__joint_axis_np
        ret.joint_origins = self.__joint_origins_np
        ret.pos_lim = self.__pos_lim_np
        ret.vel_lim = self.__vel_lim_np
        ret.eff_lim = self.__eff_lim_np
        ret.continuous_joints = self.__continuous_joints_np
        ret.pos_lim_mask = self.__pos_lim_mask_np

        # update joint and link data
        ret.joint_data = {k:v.numpy() for k,v in self.joint_data.items()}
        ret.link_data = {k:v.numpy() for k,v in self.link_data.items()}

        # save the device parameter
        ret.device = None
        ret.dtype = self.__basetype.numpy().dtype
        return ret
    
    def to(self, device=None):
        # create a shallow copy of self
        ret = copy.copy(self)
        # update references to all the properties we care about
        ret.joint_axis = self.__joint_axis.to(device=device)
        ret.joint_origins = self.__joint_origins.to(device=device)
        ret.pos_lim = self.__pos_lim.to(device=device)
        ret.vel_lim = self.__vel_lim.to(device=device)
        ret.eff_lim = self.__eff_lim.to(device=device)
        ret.continuous_joints = self.__continuous_joints.to(device=device)
        ret.pos_lim_mask = self.__pos_lim_mask.to(device=device)

        # update joint and link data
        ret.joint_data = {k:v.to(device=device) for k,v in self.joint_data.items()}
        ret.link_data = {k:v.to(device=device) for k,v in self.link_data.items()}

        # save the device parameter
        ret.device = device
        ret.dtype = self.__basetype.dtype
        return ret


class ZonoArmRobotJoint:
    __slots__ = [
        'np',
        'device',
        'tensor',
        'axis',
        'origin',
        'radius',
        'aabb',
        'outer_pz',
        'bounding_pz',
        '_actuated_idx',
        '_joint_occupancy',
        '_axis',
        '_origin',
        '_radius',
        '_aabb',
        '_axis_np',
        '_origin_np',
        '_radius_np',
        '_aabb_np',
        '_outer_pz',
        '_bounding_pz',
        ]
    
    def __init__(self, actuated_idx, joint_occupancy):
        self._actuated_idx = actuated_idx
        self._joint_occupancy = joint_occupancy
        self.radius = None
        self.aabb = None
        self.outer_pz = None
        self.bounding_pz = None
        pass

    def numpy(self):
        # create a shallow copy of self
        ret = copy.copy(self)
        # update references to all the properties we care about
        ret.axis = self._axis_np
        ret.origin = self._origin_np
        if self._joint_occupancy:
            ret.radius = self._radius_np
            ret.aabb = self._aabb_np
            ret.outer_pz = None
            ret.bounding_pz = None

        # Save the device
        ret.device = None
        return ret

    def to(self, device=None):
        # create a shallow copy of self
        ret = copy.copy(self)
        # update references to all the properties we care about
        ret.axis = self._axis.to(device=device)
        ret.origin = self._origin.to(device=device)
        if self._joint_occupancy:
            ret.radius = self._radius.to(device=device)
            ret.aabb = self._aabb.to(device=device)
            ret.outer_pz = self._outer_pz.to(device=device)
            ret.bounding_pz = self._bounding_pz.to(device=device)

        # Save the device
        ret.device = device
        return ret
    
    @property
    def idx(self):
        return self._actuated_idx
    

class ZonoArmRobotLink:
    __slots__ = [
        'np',
        'device',
        'tensor',
        'bounding_pz',
        'mass',
        'center_mass',
        'inertia',
        '_bounding_pz',
        '_mass',
        '_center_mass',
        '_inertia',
        '_mass_np',
        '_center_mass_np',
        '_inertia_np',
        ]
    
    def __init__(self):
        self.bounding_pz = None
        pass

    def numpy(self):
        # create a shallow copy of self
        ret = copy.copy(self)
        # update references to all the properties we care about
        ret.bounding_pz = None
        ret.mass = self._mass_np
        ret.center_mass = self._center_mass_np
        ret.inertia = self._inertia_np

        # Save the device
        ret.device = None
        return ret

    def to(self, device=None):
        # create a shallow copy of self
        ret = copy.copy(self)
        # update references to all the properties we care about
        ret.bounding_pz = self._bounding_pz.to(device=device)
        ret.mass = self._mass.to(device=device)
        ret.center_mass = self._center_mass.to(device=device)
        ret.inertia = self._inertia.to(device=device)

        # Save the device
        ret.device = device
        return ret


def _generate_joint_occupancy(robot, joint, DEBUG_VIZ=False):
    # This assumes the home position is aligned.
    # This assumes simple chain kinematics, no branching has been considered
    # Iterate all joints
    # This only returns the interval, but could be modified to get a ball or similar.
    # Get the trimesh and transform for the collision for multiple configurations
    if joint.joint_type in ['prismatic', 'floating', 'planar']:
        raise NotImplementedError
    # Get the links we care about too
    parent_link = robot.link_map[joint.parent]
    child_link = robot.link_map[joint.child]
    
    # FLag if we are going to treat it as a fixed joint
    # For the fixed joint, we want intersections of three orthogonal rotations.
    # So to acheive that, we just add one more orthogonal rotation.
    treat_fixed = None

    # Do an initial union and bounding box.
    res = robot.collision_trimesh_fk(links=[parent_link, child_link])
    # if we only have one mesh, rotate 180 and then treat it as fixed
    if len(res) != 2:
        basis = normal_vec_to_basis(joint.axis)
        rotmat = np.hstack(([0,0,0],basis[:,0])) * np.pi
        rotmat = xyz_rpy_to_matrix(rotmat)
        first_el = list(res.items())[0]
        res[first_el[0].copy().apply_transform(rotmat)] = first_el[1]
        child_volume = first_el[0].bounding_box.volume
        treat_fixed = True
    else:
        child_volume = child_link.collision_mesh.bounding_box.volume
    res = [mesh.copy().apply_transform(transform) for mesh, transform in res.items()]
    combined_mesh = trimesh.util.concatenate(res[0].bounding_box, res[1].bounding_box)
    combined_mesh_debug = trimesh.util.concatenate(res[0], res[1]) if DEBUG_VIZ else None
    base_volume = combined_mesh.bounding_box.volume
    child_volume_ratio = child_volume / base_volume

    # Move and union the bounding boxes for links in the joint if it's moveable
    if joint.joint_type in ['revolute', 'continuous'] and not treat_fixed:
        for ang in JOINT_MOTION_UNION_ANGLES:
            res = robot.collision_trimesh_fk(cfg={joint.name: ang}, links=[child_link])
            res = [mesh.copy().apply_transform(transform) for mesh, transform in res.items()]
            combined_mesh = trimesh.util.concatenate(combined_mesh, res[0].bounding_box)
        # Check if it was mostly inline by taking the ratio of added volume to that of a child link
        added_ratio = combined_mesh.bounding_box.volume / base_volume - 1
        ratio = added_ratio / child_volume_ratio
        treat_fixed = (ratio < INLINE_RATIO_CUTOFF)
    
    # Get the child base transform, and invert to get the joint at the origin
    res = robot.link_fk(links=[child_link])
    base_transform = list(res.values())[0]

    combined_mesh.apply_transform(np.linalg.pinv(base_transform))
    if DEBUG_VIZ:
        combined_mesh_debug.apply_transform(np.linalg.pinv(base_transform))

    ###
    # This section didn't work right, and was unreliable.
    ###
    # Do the rotations (create a transform for 90 degree rotations)
    # trimmed_mesh = combined_mesh.copy().bounding_box
    # rot_transform = np.hstack(([0,0,0], joint.axis)) * np.pi * 0.5
    # rot_transform = xyz_rpy_to_matrix(rot_transform)
    # for _ in range(JOINT_INTERSECT_COUNT):
    #     rot_mesh = trimmed_mesh.copy().apply_transform(rot_transform)
    #     trimmed_mesh = trimesh.boolean.boolean_automatic([trimmed_mesh, rot_mesh], 'intersection').bounding_box

    # # Do additional orthogonal rotations of the trimed mesh if it is fixed, or we are treating it as fixed
    # if joint.joint_type == 'fixed' or treat_fixed:
    #     basis = normal_vec_to_basis(joint.axis)
    #     xyzrpy = np.hstack((np.zeros((2, 3)), basis[:,:2].T)) * np.pi * 0.5
    #     for transform in xyzrpy:
    #         rot_transform = xyz_rpy_to_matrix(transform)
    #         # Adding bounding_box here addresses a nefarious bug where it doesn't actually create
    #         # valid geometry for blender to work with.
    #         rot_mesh = trimmed_mesh.copy().apply_transform(rot_transform).bounding_box
    #         trimmed_mesh = trimesh.boolean.boolean_automatic([trimmed_mesh, rot_mesh], 'intersection').bounding_box

    # Instead, since we're now centered on the joint, just obtain the min and max of the
    # absolute values of each vector element
    aabb_verts = combined_mesh.bounding_box.vertices
    extents = np.abs(aabb_verts)
    extents = np.column_stack([np.amin(extents, axis=0), np.amax(extents, axis=0)])

    # For the dimensions coplaner to the plane described by the axis normal, we take the min of the extends
    basis = normal_vec_to_basis(joint.axis)
    # Right now I only consider axis aligned situations, so throw for other cases.
    if not np.all(np.any(basis==1, axis=0)):
        raise NotImplementedError
    # Isolate the off axis components & take the combined min of the absolute extents
    _, basis_order = np.nonzero(basis.T)
    joint_bb = np.zeros((3,2))
    joint_bb[basis_order[:2],1] = np.min(extents[basis_order[:2],0])
    joint_bb[basis_order[:2],0] = -joint_bb[basis_order[:2],1]
    # If it's something we treat as fixed, compare to the min of the on-axis components
    # and make that it
    if joint.joint_type == 'fixed' or treat_fixed:
        on_axis = np.min(extents[:2,0])
        joint_bb[basis_order[2],:] = [-on_axis, on_axis]
    else:
        # For the on-axis component, take the aabb_bounds
        aabb_bounds = np.column_stack([np.amin(aabb_verts, axis=0), np.amax(aabb_verts, axis=0)])
        joint_bb[basis_order[2],:] = aabb_bounds[basis_order[2],:]

    if DEBUG_VIZ:
        print("Treat Fixed: ", treat_fixed)
        mesh = combined_mesh_debug
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)
        mesh = trimesh.primitives.Box(bounds=joint_bb.T)
        ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, alpha=0.2)
        interval_size = np.max(np.abs(joint_bb), axis=1)
        mesh = trimesh.primitives.Box(bounds=[-interval_size, interval_size])
        ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, alpha=0.2)
        plt.show()

    # This doesn't create the right bounds for the last one, but that's okay for now.
    # out_dict = {}
    # out_dict['radius'] = joint_bb[:,1]
    # out_dict['aabb'] = joint_bb

    # Store zonotopes for the outer_bb and the aabb
    # outer_rad = np.max(out_dict['radius'])
    # out_dict['outer_pz'] = zp.polyZonotope(torch.vstack([torch.zeros(3), torch.eye(3)*outer_rad]))
    # center = np.sum(out_dict['aabb'], axis=1) / 2
    # gens = np.diag(out_dict['aabb'][:,1] - center)
    # out_dict['bounding_pz'] = zp.polyZonotope(torch.as_tensor(np.vstack([center,gens]), dtype=torch.get_default_dtype()))
    # return out_dict

    radius = joint_bb[:,1]
    aabb = joint_bb
    return radius, aabb

if __name__ == '__main__':
    import os
    import zonopy as zp
    basedirname = os.path.dirname(zp.__file__)
    a = load_robot(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'))
    a.show()

    link = a.links[4]
    mesh = link.collision_mesh
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)

    mesh = link.collision_mesh.bounding_box
    # fig=plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, alpha=0.2)

    mesh = link.collision_mesh.bounding_box_oriented
    fig=plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)

    mesh = link.collision_mesh.bounding_cylinder
    fig=plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, alpha=0.2)

    test = a.collision_trimesh_fk(cfg={'joint_4':3.14/2.0},links=[link])
    mesh = link.collision_mesh
    from trimesh import transformations
    mesh.apply_transform(transformations.random_rotation_matrix())
    fig=plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces)

    mesh = link.collision_mesh.bounding_box
    import matplotlib.pyplot as plt
    # fig=plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(mesh.vertices[:,0],mesh.vertices[:,1], mesh.vertices[:,2], triangles=mesh.faces, alpha=0.2)

    plt.show()

    print('end')
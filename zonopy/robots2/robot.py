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


class ArmRobot:
    # __slots__ = ['mass', '', 'G']
    def __init__(self, robot: Union[URDF, str]):
        if type(robot) == str:
            robot = load_robot(robot)
        self.robot = robot
        
        self.num_q = len(robot.actuated_joints)

        self.joint_axis = [joint.axis for joint in robot.actuated_joints]
        self.joint_axis = np.array(self.joint_axis)

        self.joint_origins = [joint.origin for joint in robot.actuated_joints]
        self.joint_origins = np.array(self.joint_origins)

        self.actuated_joint_names = [joint.name for joint in robot.actuated_joints]
        

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
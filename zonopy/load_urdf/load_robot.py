from zonopy.load_urdf.urdf_parser_py.urdf import URDF
from zonopy.load_urdf.tree import rigidBodyTree
from zonopy.load_urdf.utils import parellel_axis
import torch
import os
import copy

from zp import polyZonotope, matPolyZonotope
dirname = os.path.dirname(__file__)
ROBOTS_PATH = os.path.join(dirname,'assets/robots')

ROBOT_ARM_PATH = {'fetch': 'fetch_arm/fetch_arm.urdf',
                  'kinova3': 'kinova_arm/gen3.urdf',
                  'kuka': 'kuka_arm/kuka_iiwa_arm.urdf',
                  'ur5': 'ur5_robot.urdf'
                  }

JOINT_TYPE_MAP = {'revolute': 'revolute',
                  'continuous': 'revolute',
                  'prismatic': 'prismatic',
                  'fixed': 'fixed'}

def import_robot(urdf_file,gravity=True):
    if urdf_file in ROBOT_ARM_PATH.keys():
        urdf_file = ROBOT_ARM_PATH[urdf_file]
        urdf_path = os.path.join(ROBOTS_PATH,urdf_file)
    else:
        urdf_path = urdf_file
    # parse URDF file
    robot_urdf = URDF.from_xml_string(open(urdf_path).read())
    # order link and joints
    link_map = robot_urdf.link_map
    joints = robot_urdf.joints
    n_joints = len(joints)
    has_root = [True for _ in range(n_joints)]
    for i in range(n_joints):
        for j in range(i+1,n_joints):
            if joints[i].parent == joints[j].child:
                has_root[i] = False
            elif joints[j].parent == joints[i].child:
                has_root[j] = False
    
    for i in range(n_joints):
        if has_root[i]:
            base = link_map[joints[i].parent]
    robot = rigidBodyTree(robot_urdf.links,joints,base)
    import pdb; pdb.set_trace()
    return robot

def load_sinlge_robot_arm_params(urdf_file,gravity=True):
    '''
    Assumed all the active joint is revolute.
    '''
    robot = import_robot(urdf_file,gravity)

    params = {}
    mass = [] # mass
    I = [] # momnet of inertia
    G = [] # spatial inertia
    com = [] # CoM position
    com_rot = [] # CoM orientation
    joint_axes = [] # joint axes
    H = [] # transform of ith joint in pre. joint in home config.
    R = [] # rotation of    "   "
    P = [] # translation of    "   "  
    M = [] # transform of ith CoM in prev. CoM in home config.
    screw = [] # screw axes in base
    pos_lim = [] # joint position limit
    vel_lim = [] # joint velocity limit
    tor_lim = [] # joint torque limit

    Tj = torch.eye(4,dtype=torch.float) # transform of ith joint in base
    K = torch.eye(4,dtype=torch.float) # transform of ith CoM in ith joint
    
    body = robot[robot.base.children_id[0]]
    for i in range(robot.n_bodies):
        mass.append(body.mass)
        I.append(body.inertia)
        #import pdb; pdb.set_trace()
        G.append(torch.block_diag(body.inertia,body.mass*torch.eye(3)))
        com.append(body.com)
        com_rot.append(body.com_rot)
        joint_axes.append(body.joint.axis)
        H.append(body.transform)
        R.append(body.transform[:3,:3])
        P.append(body.transform[:3,3])

        Tj = Tj @ body.transform # transform of ith joint in base
        K_prev = K
        K = torch.eye(4)
        K[:3,:3],K[:3,3] = body.com_rot, body.com 
        M.append(torch.inverse(K_prev)@body.transform@K)

        w = Tj[:3,:3] @ body.joint.axis
        v = torch.cross(-w,Tj[:3,3])
        screw.append(torch.hstack((w,v)))
        
        pos_lim.append(body.joint.pos_lim)
        vel_lim.append(body.joint.vel_lim)
        tor_lim.append(body.joint.f_lim)
        
        if len(body.children_id)>1 or robot[body.children_id[0]].joint.type == 'fixed':
            n_joints = i+1
            break
        else:
            body = robot[body.children_id[0]]

    params = {'mass':mass, 'I':I, 'G':G, 'com':com, 'com_rot':com_rot, 'joint_axes':joint_axes,
    'H':H, 'R':R, 'P':P, 'M':M, 'screw':screw,
    'pos_lim':pos_lim, 'vel_lim':vel_lim, 'tor_lim':tor_lim,
    'n_bodies': robot.n_bodies, 'n_joints': n_joints,
    'gravity': robot.gravity, 'use_interval': False
    }
    return params, robot


def load_inerval_params(urdf_file,use_random=True,gravity=True,robot_type='single_arm'):
    if robot_type == 'single_arm':
        true_params,true_robot = load_sinlge_robot_arm_params(urdf_file,gravity)
        hi_robot, lo_robot = copy.deepcopy(true_robot), copy.deepcopy(true_robot)
        nominal_params,interval_params = true_params.copy(), true_params.copy()
    else:
        assert False, 'Invalid robot type.'
    
    n_joints = interval_params['n_joints']
    lo_mass_scale, hi_mass_scale = 0.97, 1.03
    lo_com_scale, hi_com_scale = 1,1
    lo_mass, hi_mass = torch.zeros(n_joints), torch.zeros(n_joints)
    lo_com, hi_com = torch.zeros(n_joints,3), torch.zeros(n_joints,3)

    I = []
    G = []
    for i in range(n_joints+1):
        # scale link masses
        lo_robot[i].mass *= lo_mass_scale
        hi_robot[i].mass *= hi_mass_scale
        # scale center of mass
        lo_robot[i].com *= lo_com_scale
        hi_robot[i].com *= hi_com_scale
        # scale inertia
        lo_robot[i].inertia *= lo_mass_scale
        hi_robot[i].inertia *= hi_mass_scale


        lo_com, hi_com = torch.min(lo_robot[i].com,hi_robot[i].com), torch.max(lo_robot[i].com,hi_robot[i].com)

        lo_I = parellel_axis(lo_robot[i].inertia, lo_robot[i].mass, lo_robot[i].com_rot, lo_robot[i].com)
        hi_I = parellel_axis(hi_robot[i].inertia, hi_robot[i].mass, hi_robot[i].com_rot, hi_robot[i].com)
        lo_I, hi_I = torch.min(lo_I,hi_I), torch.max(lo_I,hi_I)
        
        

    return true_params, nominal_params, interval_params, lo_robot,hi_robot




def load_poly_zono_params(urdf_file,use_random=True,gravity=True,robot_type='single_arm'):
    if robot_type == 'single_arm':
        true_params,true_robot = load_sinlge_robot_arm_params(urdf_file,gravity)
        hi_robot, lo_robot = copy.deepcopy(true_robot), copy.deepcopy(true_robot)
        nominal_params,poly_zono_params  = true_params.copy(), true_params.copy()
    else:
        assert False, 'Invalid robot type.'
    
    n_joints = poly_zono_params['n_joints']
    poly_zono_params['type'] = polyZonotope
    
    lo_mass_scale, hi_mass_scale = 0.97, 1.03
    lo_com_scale, hi_com_scale = 1,1
    lo_mass, hi_mass = torch.zeros(n_joints), torch.zeros(n_joints)
    lo_com, hi_com = torch.zeros(n_joints,3), torch.zeros(n_joints,3)

    I = []
    G = []
    mass = []
    com = []
    for i in range(n_joints+1):
        # scale link masses
        lo_robot[i].mass *= lo_mass_scale
        hi_robot[i].mass *= hi_mass_scale
        lo_mass, hi_mass = lo_robot[i].mass, hi_robot[i].mass
        C = (lo_mass+hi_mass)/2*torch.eye(3)
        G = (hi_mass-lo_mass)/2*torch.eye(3)
        # NOTE: update ID
        mass.append(matPolyZonotope(C,G))

        # scale center of mass
        lo_robot[i].com *= lo_com_scale
        hi_robot[i].com *= hi_com_scale
        lo_com, hi_com = torch.min(lo_robot[i].com,hi_robot[i].com), torch.max(lo_robot[i].com,hi_robot[i].com)
    
        c = (lo_com+hi_com)/2
        g = torch.diag((hi_com-lo_com)/2)
        # NOTE: update ID
        com.append(polyZonotope(c,g))

        # scale inertia
        lo_robot[i].inertia *= lo_mass_scale
        hi_robot[i].inertia *= hi_mass_scale

        lo_I = parellel_axis(lo_robot[i].inertia, lo_robot[i].mass, lo_robot[i].com_rot, lo_robot[i].com)
        hi_I = parellel_axis(hi_robot[i].inertia, hi_robot[i].mass, hi_robot[i].com_rot, hi_robot[i].com)
        lo_I, hi_I = torch.min(lo_I,hi_I), torch.max(lo_I,hi_I)
        
        C = (lo_I+hi_I)/2*torch.eye(3)
        G
        

    return true_params, nominal_params, interval_params, lo_robot,hi_robot


if __name__ == '__main__':
    #import_robot('fetch')

    load_sinlge_robot_arm_params('fetch')


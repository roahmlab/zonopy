import json
from zonopy.conSet import interval, polyZonotope, matPolyZonotope
from torch import Tensor
import torch
def parallel_axis(inertia_vec, mass, com):
    '''
    %% PARALLEL AXIS
    % Inputs
    %   inertia_vec - the inertia tensor obtained from RigidBody object
    %   mass - mass of the link
    %   com - the center of mass relative to the joint frame
    %
    % Note that RigidBody expresses the inertia tensor with respect to 
    % the joint frame. This function uses the parallel-axis theorem to 
    % compute the inertia tensor in the link body frame whose origin is
    % located at the link center of mass. This function returns the inertia
    % values that are in the URDF.
    
    % We are computing the following:
    % I_{xx}^{C} = I_{xx}^{A} - m (y_c^2 + z_c^2);
    % I_{yy}^{C} = I_{yy}^{A} - m (x_c^2 + z_c^2);
    % I_{zz}^{C} = I_{zz}^{A} - m (x_c^2 + y_c^2);
    %
    % I_{xy}^{C} = I_{xy}^{A} + m x_c y_c;
    % I_{xz}^{C} = I_{xz}^{A} + m x_c z_c;
    % I_{yz}^{C} = I_{yz}^{A} + m y_c z_c;
    '''

    ## compute new inertia tensor
    # mass
    m = mass

    # center of mass coordinates
    x, y, z = com[0].item(), com[1].item(), com[2].item() 

    # inertia vec from matlab rigidBody object
    Ixx = inertia_vec[0]
    Iyy = inertia_vec[1]
    Izz = inertia_vec[2]
    Iyz = inertia_vec[3]
    Ixz = inertia_vec[4]
    Ixy = inertia_vec[5]
    
    # parallel axis theorem (see Craig 6.25)
    Ixx = Ixx - m * (y ** 2 + z ** 2)
    Iyy = Iyy - m * (x ** 2 + z ** 2)
    Izz = Izz - m * (x ** 2 + y ** 2)
    
    Ixy, Ixz, Iyz = Ixy+m*x*y, Ixz+m*x*z, Iyz+m*y*z
    
    I = Tensor([[Ixx, Ixy, Ixz],
         [Ixy, Iyy, Iyz],
         [Ixz, Iyz, Izz]])

    return I
    
def get_robot_params(param_file):
    with open(param_file) as f:
        param = json.load(f)
    return param

def get_interval_params(param_file):
    with open(param_file) as f:
        params = json.load(f)
    nominal_params = params.copy()
    interval_params = params.copy()

    num_joints = interval_params['num_joints']
    
    I = {}
    lo_mass_scale, hi_mass_scale = 0.97, 1.03
    lo_com_scale, hi_com_scale = 1, 1
    lo_mass,hi_mass = torch.zeros(1,num_joints), torch.zeros(1,num_joints)
    lo_com,hi_com = torch.zeros(3,num_joints), torch.zeros(3,num_joints)

    mass = param['mass']
    com = Tensor(param['com'])

    for i in range(num_joints):
        lo_inertial_vec = Tensor(param['inertia'][i])
        hi_inertial_vec = Tensor(param['inertia'][i])

        lo_I = parallel_axis((lo_mass_scale * lo_inertial_vec).tolist(), lo_mass_scale * mass[i], com[:,i])
        hi_I = parallel_axis((hi_mass_scale * hi_inertial_vec).tolist(), hi_mass_scale * mass[i], com[:,i])

        for j in range(3):
            for k in range(3):
                a = min(lo_I[j,k], hi_I[j,k]).clone()
                b = max(lo_I[j,k], hi_I[j,k]).clone()

                lo_I[j,k] = a
                hi_I[j,k] = b

        I.append(interval(Tensor(lo_I), Tensor(hi_I)))

    mass = Tensor(param['mass'])
    com = Tensor(param['com'])
    param['mass'] = interval(lo_mass_scale * mass, hi_mass_scale * mass)
    param['com'] = interval(com.clone(), com.clone())
    param['I'] = I
    param['use_interval'] = True
    return param

if __name__ == '__main__':
    param_file = "fetch_arm_param.json"
    param = get_interval_params(param_file)
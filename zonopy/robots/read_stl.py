import torch
import struct
from zonopy.robots.urdf_parser_py.urdf import URDF
import os

ROBOT_ARM_PATH = {'Fetch': 'fetch_arm/fetch_arm_reduced.urdf',
                  'Kinova3': 'kinova_arm/gen3.urdf',
                  'Kuka': 'kuka_arm/kuka_iiwa_arm.urdf',
                  'UR5': 'ur5_robot.urdf'
                  }

dirname = os.path.dirname(__file__)
ROBOTS_PATH = os.path.join(dirname,'assets/robots')
urdf_path = os.path.join(ROBOTS_PATH,ROBOT_ARM_PATH['Fetch'])
robot_urdf = URDF.from_xml_string(open(urdf_path,'rb').read())

# robot_urdf.links[i].collision.origin. (xyz or rpy)
# robot_urdf.links[i].collision.origin. (xyz or rpy)
# robot_urdf.links[i].collision.geometry (filename or scale)

import pdb; pdb.set_trace()
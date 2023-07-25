from zonopy.conSet.zonotope.zono import zonotope
from zonopy.conSet.zonotope.mat_zono import matZonotope
from zonopy.conSet.polynomial_zonotope.poly_zono import polyZonotope
from zonopy.conSet.polynomial_zonotope.mat_poly_zono import matPolyZonotope

from zonopy.conSet.zonotope.batch_zono import batchZonotope
from zonopy.conSet.zonotope.batch_mat_zono import batchMatZonotope
from zonopy.conSet.polynomial_zonotope.batch_poly_zono import batchPolyZonotope
from zonopy.conSet.polynomial_zonotope.batch_mat_poly_zono import batchMatPolyZonotope

from zonopy.conSet.interval.interval import interval

from zonopy.joint_reachable_set.jrs_trig.load_jrs_trig import preload_batch_JRS_trig,load_JRS_trig, load_traj_JRS_trig, load_batch_JRS_trig, load_batch_JRS_trig_ic
from zonopy.joint_reachable_set.jrs_trig.process_jrs_trig import process_batch_JRS_trig
from zonopy.joint_reachable_set.gen_jrs import JrsGenerator

from zonopy.utils import *
from zonopy.robots.load_robot import *

__version__ = "0.0.1"
__logo__ = """
*** ZONO-PY ***
  _____
 |     |__
 |_____|  |__
    |_____|  |
       |_____|
"""

def setup_cuda():
   if torch.cuda.is_available():
      #self.__device =f'cuda:{device_num}'
      #torch.cuda.set_device(device_num)
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
      from zonopy.conSet import DEFAULT_OPTS
      DEFAULT_OPTS.set(device='cuda')
   else:
      print('WARNING: No CUDA GPUs are available.')


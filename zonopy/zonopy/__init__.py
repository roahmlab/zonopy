from zonopy.contset import *
from zonopy.util import *
from zonopy.math import *

import zonopy.internal as internal

__version__ = "0.1.0"
__logo__ = """
*** ZONO-PY ***
  _____
 |     |__
 |_____|  |__
    |_____|  |
       |_____|
"""

def setup_cuda():
   if not internal.__hide_deprecation__:
      import warnings
      warnings.warn(
         'This function is deprecated. Declare devices and dtypes in the constructor.',
         DeprecationWarning)
   import torch
   if torch.cuda.is_available():
      #self.__device =f'cuda:{device_num}'
      #torch.cuda.set_device(device_num)
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
      from zonopy.contset import DEFAULT_OPTS
      DEFAULT_OPTS.set(device='cuda')
   else:
      print('WARNING: No CUDA GPUs are available.')

from zonopy import polyZonotope, matPolyZonotope, batchPolyZonotope, batchMatPolyZonotope
from collections import OrderedDict
from .FK import forward_kinematics
from urchin import URDF

from typing import Union, Dict, List, Tuple
from typing import OrderedDict as OrderedDictType

# Use forward kinematics to get the forward occupancy
def forward_occupancy(rotatotopes: Union[Dict[str, Union[matPolyZonotope, batchMatPolyZonotope]],
                                         List[Union[matPolyZonotope, batchMatPolyZonotope]]],
                      robot: URDF,
                      zono_order: int = 20,
                      links: List[str] = None,
                      link_zono_override: Dict[str, polyZonotope] = None,
                      ) -> Tuple[OrderedDictType[str, Union[polyZonotope, batchPolyZonotope]],
                                 OrderedDictType[str, Union[Tuple[polyZonotope, matPolyZonotope],
                                                            Tuple[batchPolyZonotope, batchMatPolyZonotope]]]]:
    
    link_fk_dict = forward_kinematics(rotatotopes, robot, zono_order, links=links)
    link_zonos = {name: robot._link_map[name].bounding_pz for name in link_fk_dict.keys()}
    if link_zono_override is not None:
        link_zonos.update(link_zono_override)
    
    fo = OrderedDict()
    for name, (pos, rot) in link_fk_dict.items():
        link_zono = link_zonos[name]
        fo_link = pos + rot@link_zono
        fo[name] = fo_link.reduce_indep(zono_order)
    
    return fo, link_fk_dict
        
import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import numpy as np

from module.Hamiltonian.Hamiltonian4TB.tight_binding_hamiltonian import TBHamiltonian
from module.physics_system.periodicity.Periodicity_System import PeriodicityPhysicsSystem
# model_input = {"sysinit":{
#                             "sys_name":"Si_sps'",
#                             "group_type":"Space Group",
#                             "group_name":"227",
#                             "lattice_type":"CubiFace",
#                             "lattice_parameter":{"a":1},
#                             "atompos":[[1/8,1/8,1/8]],
#                             # "magdirect":[[0,0,0]],
#                             "n_neighbour":2
#                             },
#             "orbit_init":[{"orbit_list":["s","s","px","py","pz"]}]}

sys_init = {
    "sys_name":"MX2",
    "group_type":"Magnetic Group",
    "group_name":"14.84",
    "lattice_type":"MonoPrim",
    "lattice_parameter":{"a":1,"Gamma":96/180*np.pi,"b":1,"c":3},
    # "atompos":[[0.5,0.75,0.75],[0.5,0.5,0],[0.5,0.25,0.25],[0.46,0.68,0.0],[0.54,0.5,0.18],[0.46,0.18,0],[0.54,0.32,0.57]],
    # "magdirect":[[0,0,1],[0,0,-1],[0,0,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    "atompos":[[0.5,0,0]],
    "magdirect":[[0,0,-1]],
    "n_neighbour":2
}
system = PeriodicityPhysicsSystem(**sys_init)
print(system.wyckoffpos)
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
    "group_name":"14.75",
    "lattice_type":"MonoPrim",
    "lattice_parameter":{"a":5,"Gamma":90/180*np.pi,"b":1,"c":1},
    "atompos":[[0.5,0,0],[0.54,0.85,0.35]],
    "magdirect":[[0,0,-1],[0,0,0]],
    "neighbour_list":[3,3]
}
orbit_init = [{"orbit_list":["dyz",],"spin_dict":{"dyz":1}},{"orbit_list":["pz","py"],"spin_dict":{"pz":0,"py":0}}]
model = TBHamiltonian(sysinit=sys_init,orbit_init=orbit_init)
print(model.sym_hamiltonian_dict)
print(model.sym_hamiltonian_dict[0].shape)
# system = PeriodicityPhysicsSystem(**sys_init)
# print(system.wyckoffpos)
# print(system.neighbour_table)
# print(len(system.atom_distance))
# print(system.atom_distance[0][2])
# print("\n________________________________________\n")
# print(system.atom_distance[0][3])
# print(len(system.neighbour_table))



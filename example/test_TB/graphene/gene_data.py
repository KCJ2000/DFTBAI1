import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import time
import torch

from module.physics_property.band.band import Band

band = Band()
para_input = torch.tensor([[0, 0.5, 0]])
band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/Graphene_pz.pkl",
                              para = para_input
                              )

n_kpoints = 20
kpoint = [[0,0,0],[0,1/2,0],[1/3,1/3,0],[0,0,0]]
klabel = ["GAMMA","M","K","GAMMA"]

band.calculate_band(kpoints=kpoint,klabels=klabel,nkpoints=n_kpoints)
# print(band.content['energy'])
band.save_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/band.npz")
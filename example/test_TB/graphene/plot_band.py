import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import torch


from module.physics_property.band.band import Band

band = Band()
para_input = torch.tensor([[-2.6058e-04,  0.5, -1.0008e-04]])
band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/Graphene_pz.pkl",
                              para = para_input
                              )
band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/band.npz")
# print(band.content['k_vector'])
model_index = [0,1]
band_index = [0,1]
# band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/calculate_band.png",
#                     select_band=select_band
#                     )
band.plot_compare(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/compare.png",
                  model_index=model_index,band_index=band_index,title="graphene")
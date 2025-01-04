import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import torch


from module.physics_property.band.band import Band

band = Band()
para_input = torch.tensor([[-0.0121, -0.5198,  0.0132]])
band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/Graphene_pz.pkl",
                              para = para_input
                              )
band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/band.npz")
print(band.content['k_vector'].shape)
select_band = [0,1]
band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/graphene/calculate_band.png",
                    select_band=select_band
                    )
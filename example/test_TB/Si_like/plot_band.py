import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import torch


from module.physics_property.band.band import Band

band = Band()
para_input = torch.tensor([[ 0.2866,  0.0000,  0.0000, 11.8229,  1.1258, -0.6376,  0.0000,  0.0000,  0.6004,  0.0000,  0.0000,  1.6631,  0.0000, -0.2360,  2.4244]])
para_input = torch.tensor([[ 5.4301, -2.3434, -7.8491,  3.8092,  1.1535, -0.9445, -1.6489, -2.3154,  0.4516,  0.2547, -1.3107, -1.3938,  1.7186,  0.2979,  1.0931]])
para_input = torch.tensor([[-4.4515,  0.0000,  0.0000,  7.0634,  1.2606, -1.8478,  0.0000,  0.0000, -1.2398,  0.0000,  0.0000,  1.2246,  0.0000,  0.3048,  1.1528]])
band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'.pkl",
                              para = para_input
                              )
band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
print(band.content['k_vector'].shape)
select_band = [1,2,3,4]
band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/calculate_band.png",
                    select_band=select_band
                    )
import os
import sys
file_path = sys.argv[0]
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path,"module"))
import torch


from module.physics_property.band.band import Band

band = Band()
para_input = torch.tensor([[-4.6034,  0.0000,  0.0000,  5.1515,  1.3112,  1.9006,  0.0000,  0.0000,  1.2579,  0.0000,  0.0000,  0.9963,  0.0000, -0.2933, -1.1782]])
para_input = torch.tensor([[-4.2,0,0,6.685,1.7,-8.3/4,0,0,-5.7292/4,0,0,-5.3749/4,0,1.715/4,4.575/4]])### data in paper
para_input = torch.tensor([[ 1.5263, -4.3383, -2.4179, -1.0207,  1.3008,  0.7829,  1.1961,  0.5175, -0.1986, -0.3115,  0.9014, -0.6172,  1.3901, -0.2847, -1.1724]])### without mask
para_input = torch.tensor([[-4.6034,  0.0000,  0.0000,  5.1515,  1.3112,  1.9006,  0.0000,  0.0000,  1.2579,  0.0000,  0.0000,  0.9963,  0.0000, -0.2933, -1.1782]])### with mask
band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'.pkl",
                              para = para_input
                              )
band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
print(band.content['k_vector'].shape)
print(band.content["energy"].shape)
select_band = [1,2,3,4]
# band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/calculate_band.png",
#                     select_band=select_band
#                     )

band.plot_compare(input_data=band.content['k_vector'],
                  save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/compare.png",
                  model_index=select_band,
                  band_index=select_band,title="Si_sps' with mask")

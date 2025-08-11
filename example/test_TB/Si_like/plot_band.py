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
para_input = torch.tensor([[-0.2691,  0.0000,  0.0000, -3.6528,  2.2731, -0.5708,  0.0000,  0.0000,
          1.2858,  0.0000,  0.0000,  0.4219,  0.0000,  0.7847, -0.3127]])
para_input = torch.tensor([[-2.1812,  0.0000,  0.0000,  7.5044,  1.4667,  1.0641,  0.0000,  0.0000,
         -1.1297,  0.0000,  0.0000, -0.9795,  0.0000, -0.3762, -1.1673]])
para_input = torch.tensor([[-4.3443,  0.0000,  0.0000,  5.4121,  2.6198, -0.8687,  0.0000,  0.0000,
         -0.3176,  0.0000,  0.0000, -1.9344,  0.0000,  0.9689, -0.3750]])

### Si_sps'_3n
para_input = torch.tensor([[ 2.0249, -2.3703, -3.7541, -2.2140,  0.7454,  0.0167, -1.4780,  0.6165,
         -0.3239, -0.0888, -2.5800, -0.1998,  0.1754,  0.3188,  0.8439,  0.1370,
         -0.0885,  0.4793, -0.0066, -0.2666,  0.0542,  0.1943,  0.3218, -0.5128,
         -0.0998,  0.0532, -0.3283,  0.0377,  0.0469, -0.2630, -0.1765, -0.0560]])
### Si_sps'_4n
para_input = torch.tensor([[ 1.5115,  2.9662, -1.4881,  3.1170,  0.3244, -1.2763,  0.4376, -0.2855,
          0.6256,  0.0966, -0.3768, -0.4189, -0.3760, -0.3795,  0.7596, -0.0526,
         -0.0604,  0.0450,  0.0611,  0.2377,  0.2341,  0.0858,  0.0759,  0.0258,
         -0.6101,  0.1287, -0.1998, -0.3972,  0.4401, -0.4101, -0.4737, -0.1297,
          0.2284, -0.1183,  0.0540, -0.2203, -0.0565, -0.0962,  0.0712, -0.1570,
         -0.1058, -0.0400, -0.2799,  0.1629, -0.0721,  0.5129, -0.0422, -0.1332,
          0.5353]])
band.init_calculate_model("/data/home/kongfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'_4n.pkl",
                              para = para_input
                              )
band.get_data("/data/home/kongfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
print(band.content['k_vector'].shape)
print(band.content["energy"].shape)
select_band = [1,2,3,4]
# band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/calculate_band.png",
#                     select_band=select_band
#                     )

band.plot_compare(input_data=band.content['k_vector'],
                  save_path="/data/home/kongfh/DFTBAI1/example/test_TB/Si_like/Si_PC/new_test.png",
                  model_index=select_band,
                  band_index=select_band,title="Si_sps'_4n")

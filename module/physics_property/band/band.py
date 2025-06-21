import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from sympy import Matrix,lambdify,eye,symbols
import matplotlib.pyplot as plt
import warnings
import torch

from physics_property.property import Property
from physics_property.band.band_data_in import BandDataIn
from physics_property.band.band_data_out import BandDataOut
from parameter.para4band.para4band import Para4Band


class Band(Property):
    def __init__(self):
        self.description = "the class used to do band data analysis"
        self.content = {}
        self.matrix_function = None
        
    def init_calculate_model(self, model_path,para):
        super().init_calculate_model(model_path)
        self.para_calculate = Para4Band(self.model_path)
        self.para_calculate.init_para(para)
        self.matrix_function = self.para_calculate.matrix_function
        
        
    def get_data(self,data_file_path):
        band_in = BandDataIn(file_path=data_file_path)
        self.content = band_in.content


    def save_data(self,save_file_path):
        band_out = BandDataOut(file_path=save_file_path,content=self.content)
        band_out.save_content()
        
    def calculate_band(self,kpoints,klabels,nkpoints):    
        if self.matrix_function == None:
            raise AssertionError("请初始化model")
        n_k = len(kpoints)
        input_data = []
        kpath = []
        for i in range(n_k-1):
            input_data.append(torch.stack([torch.linspace(s, e, nkpoints) for s, e in zip(kpoints[i], kpoints[i+1])], dim=0))
            kpath.append([klabels[i],klabels[i+1]])
        input_data = torch.cat(input_data,dim=-1)*2*torch.pi
        input_data = input_data.to(self.para_calculate.device).to(torch.double)
        matrix = self.matrix_function(input_data)
        print(matrix.shape)
        eigens,_ = torch.linalg.eig(matrix)
        eigens = eigens.type(torch.float64)
        eigens = torch.sort(eigens,dim=-1)[0].to("cpu")
        eigens = eigens[0]
        eigens = eigens.detach().numpy()
        input_data = input_data/2/torch.pi
        input_data = input_data.transpose(dim0=-1,dim1=-2)
        input_data = input_data.to("cpu").detach().numpy()
        self.content = {"k_vector":input_data,"n_kpoint":nkpoints,"kpath":kpath,"energy":eigens}
        
        
    def plot_model(self,input_data,save_path,select_band,colour = "b",kpath=None):
        input_data = torch.tensor(input_data,dtype=torch.float64).transpose(dim0=0,dim1=1)*2*torch.pi
        input_data = input_data.to(self.para_calculate.device)
        if self.matrix_function != None:
            matrix = self.matrix_function(input_data)
            eigens,_ = torch.linalg.eig(matrix)
            eigens = eigens.type(torch.float64)
            eigens = torch.sort(eigens,dim=-1)[0].to("cpu")
            eigens = eigens.detach().numpy()
            
            x = np.linspace(0,1,eigens.shape[1])
            for i in range(eigens.shape[2]):
                if i not in select_band:
                    continue
                plt.plot(x,eigens[0,:,i],colour)
            
            if kpath:
                pass
            elif self.content != {}:
                kpath = self.content["kpath"]
            else:raise AssertionError("没有输入kpath,请输入")
            
            n_path = len(kpath)    
            path_positions = np.linspace(0, 1, n_path + 1)
            tick_labels = []
            
            for i in range(n_path):
                if kpath[i][0] == "GAMMA":
                    kpath[i][0] = '\Gamma'
                if kpath[i][1] == "GAMMA":
                    kpath[i][1] = '\Gamma'

            for i in range(n_path):
                if i == 0:
                    tick_labels.append(f"$"+kpath[i][0]+"$")
                else:
                    if kpath[i][0] == kpath[i-1][1]:
                        tick_labels.append(f"$"+kpath[i][0]+"$")
                    else:tick_labels.append(f"$"+kpath[i-1][1]+"|"+kpath[i][0]+"$")
            tick_labels.append(f"$"+kpath[-1][1]+"$")
            plt.xticks(path_positions, tick_labels, rotation=0, ha='center')
            plt.xlim(0,1)
            for x in path_positions:
                plt.axvline(x=x, color='k', linestyle='--', linewidth=1) 
            plt.savefig(save_path)
        else:
            raise AssertionError("还没初始化model,请先用init_calculate_model(model_path)初始化计算函数")
        
    
    def plot_data(self,save_path,select_band,colour="b"):
        print("k_vector.shape",self.content['k_vector'].shape)
        n_k_points = self.content["k_vector"].shape[0]
        energy = self.content["energy"][:,select_band]
        energy = energy.reshape(n_k_points,-1)
        print("select_band.shape",select_band)
        print("energy.shape",energy.shape)
        print("select_energy",energy.shape)
        n_band = energy.shape[1]
        x = np.linspace(0,1,self.content["k_vector"].shape[0])
        for band_index in range(n_band):
            plt.plot(x,energy[:,band_index],colour)
        
        kpath = self.content['kpath']
        n_path = len(kpath)
        path_positions = np.linspace(0, 1, n_path + 1)
        tick_labels = []
        for i in range(n_path):
            if i == 0:
                tick_labels.append(f"$"+kpath[i][0]+"$")
            else:
                if kpath[i][0] == kpath[i-1][1]:
                    tick_labels.append(f"$"+kpath[i][0]+"$")
                else:tick_labels.append(f"$"+kpath[i-1][1]+"|"+kpath[i][0]+"$")
        tick_labels.append(f"$"+kpath[-1][1]+"$")
        plt.xticks(path_positions, tick_labels, rotation=0, ha='center')
        plt.xlim(0,1)
        for x in path_positions:
            plt.axvline(x=x, color='k', linestyle='--', linewidth=1)
        plt.savefig(save_path)
        

    def plot_compare(self,input_data,band_index,
                     save_path,model_index,
                     title):
        ### 画DFT
        n_k_points = self.content["k_vector"].shape[0]
        energy = self.content["energy"][:,band_index]
        energy = energy.reshape(n_k_points,-1)
        n_band = energy.shape[1]
        kpath = self.content['kpath']
        x = np.linspace(0,1,self.content["k_vector"].shape[0])
        for band_index in range(n_band):
            scatter = plt.scatter(x,energy[:,band_index],label='Hollow Circles', facecolors='none', edgecolors='r', s=13, linewidth=1.5)
        
        ### 画model
        input_data = torch.tensor(input_data,dtype=torch.float64).transpose(dim0=0,dim1=1)*2*torch.pi
        input_data = input_data.to(self.para_calculate.device)
        if self.matrix_function != None:
            matrix = self.matrix_function(input_data)
            eigens,_ = torch.linalg.eig(matrix)
            eigens = eigens.type(torch.float64)
            eigens = torch.sort(eigens,dim=-1)[0].to("cpu")
            eigens = eigens.detach().numpy()
            for i in range(eigens.shape[2]):
                if i not in model_index:
                    continue
                line, = plt.plot(x,eigens[0,:,i],"b")     
            
            n_path = len(kpath)
            path_positions = np.linspace(0, 1, n_path + 1)
            tick_labels = []
            
            for i in range(n_path):
                if kpath[i][0] == "GAMMA":
                    kpath[i][0] = "\Gamma"
                if kpath[i][1] == "GAMMA":
                    kpath[i][1] = "\Gamma"
            
            for i in range(n_path):
                if i == 0:
                    tick_labels.append(r"$"+kpath[i][0]+"$")
                else:
                    if kpath[i][0] == kpath[i-1][1]:
                        tick_labels.append(r"$"+kpath[i][0]+"$")
                    else:tick_labels.append(r"$"+kpath[i-1][1]+"|"+kpath[i][0]+"$")
            tick_labels.append(kpath[-1][1])
            plt.xticks(path_positions, tick_labels, rotation=0, ha='center')
            plt.xlim(0,1)
            for x in path_positions:### 画竖直虚线
                plt.axvline(x=x, color='k', linestyle='--', linewidth=1)
        
        plt.ylabel(r"$E_g(eV)$")
        plt.legend(handles=[line, scatter],labels=["TB model","DFT"])
        plt.title(title)
        plt.savefig(save_path)
    
    
if __name__ == "__main__":
    band = Band()
    para_input = torch.tensor([[-2.6193,  0.0000,  0.0000,  3.8806,  2.3433, -0.9512,  0.0000,  0.0000,  -1.5930,  0.0000,  0.0000, -0.0275,  0.0000, -0.0144,  1.5422]])
    para_input = torch.tensor([[-4.0771,  0.0000,  0.0000,  3.6879,  1.2270, -1.6666,  0.0000,  0.0000,  -1.4248,  0.0000,  0.0000, -0.4334,  0.0000,  0.3119,  1.1078]])
    para_input = torch.tensor([[-4.2,0,0,6.6850,1.715,-8.3/4,0,0,-5.7292/4,0,0,-5.3749/4,0,1.715/4,4.575/4]],dtype=torch.float32)
    para_input = torch.tensor([[ 3.8247,  0.0000,  0.0000,  0.9955,  0.6141,  0.4761,  0.0000,  0.0000, 0.9074,  0.0000,  0.0000,  2.0318,  0.0000,  0.1636, -1.0933]])
    para_input = torch.tensor([[-2.0294,  0.0000,  0.0000,  2.5632,  0.6566, -1.5844,  0.0000,  0.0000, 1.6382,  0.0000,  0.0000, -0.5937,  0.0000, -0.1740,  1.0706]])
    para_input = torch.tensor([[ 3.3402,  0.0000,  0.0000,  4.3304,  1.5183,  0.8866,  0.0000,  0.0000,-1.1856,  0.0000,  0.0000,  2.5173,  0.0000, -0.3892, -1.1622]])
    para_input = torch.tensor([[-4.4427,  0.0000,  0.0000, 13.8536,  1.2475,  1.9074,  0.0000,  0.0000,-1.2472,  0.0000,  0.0000, -1.8243,  0.0000, -0.3059, -1.1239]])
    para_input = torch.tensor([[-4.4251,  0.0000,  0.0000, 13.9526,  1.2214, -1.8989,  0.0000,  0.0000,-1.1981,  0.0000,  0.0000, -1.7328,  0.0000,  0.3037,  1.1151]])
    # para_input = torch.tensor([[ 1.8011e+00,  0.0000e+00,  0.0000e+00, -7.6741e-02,  3.9192e+00,9.3084e-03,  0.0000e+00,  0.0000e+00, -9.1348e-05,  0.0000e+00, 0.0000e+00, -2.4601e+00,  0.0000e+00, -9.8922e-01, -1.7848e+00]])
    # para_input = torch.tensor([[-2.1015,  0.0000,  0.0000, -0.0481,  0.5621, -1.3573,  0.0000,  0.0000,  1.6781,  0.0000,  0.0000, -0.2982,  0.0000, -0.1498,  1.0620]])
    # para_input = torch.tensor([[-3.0032,  0.0000,  0.0000,  5.7254,  1.2423, -1.3994,  0.0000,  0.0000,-1.5839,  0.0000,  0.0000, -0.9627,  0.0000,  0.3075,  1.1062]])
    # para_input = torch.tensor([[ 1.1580e+00,  0.0000e+00,  0.0000e+00,  2.9955e+00, -1.3328e+00,3.2853e-01,  0.0000e+00,  0.0000e+00, -1.5570e+00,  0.0000e+00,0.0000e+00, -2.4935e-03,  0.0000e+00, -3.1478e+00,  1.5883e+00]])
    # para_input = torch.tensor([[-6.1769e+00,  0.0000e+00,  0.0000e+00,  3.4797e-01,  4.6719e-01, -1.5343e+00,  0.0000e+00,  0.0000e+00, -9.2948e-01,  0.0000e+00, 0.0000e+00, -5.0170e-04,  0.0000e+00, -1.2592e-01,  9.8173e-01]])
    select_band = [1,2,3,4]
    band.init_calculate_model("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/Si_sps'.pkl",
                              para = para_input
                              )
    band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
    print(band.content['k_vector'].shape)
    band.plot_model(band.content['k_vector'],save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/calculate_band.png",
                    select_band=select_band
                    )
    
    # band.get_data("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
    # band.plot_data(save_path="/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/DFT_band.png",
    #                select_band=select_band)
    
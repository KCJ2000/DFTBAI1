
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from files_api.Files_in import FilesIn
import numpy as np
import os

from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.bandstructure import BandStructure

class BandDataIn(FilesIn):
    def __init__(self,file_path:str):
        super(BandDataIn,self).__init__()
        self.before_step = FilesIn()
        self.file_path = file_path
        self.description = "读取{}文件，获得band信息".format(self.file_path)
        self.content = {"k_vector":np.array([]),"energy":np.array([])}
        self.file_type = os.path.splitext(file_path)[1:][0]
        
        self.check()
        self.__module_choose()
        
        
    def check(self):
        if self.file_type == ('',):
            AssertionError("这不是一个文件路径，请输入正确路径")
        
    def __module_choose(self):
        if self.file_type == ".npz":
            self.__get_npz_content()
        elif self.file_type == ".dat":
            self.__get_dat_content()
            
    
    def __get_dat_content(self):
        """
            文件是由vaspkit生成的
        """
        def get_Kpoint(file):
            Kpoint = {}
            Kpath = []
            n_kpoint = 0
            content = file.readlines()
            n_kpoint = int(content[1])
            n_path = int((len(content)-4)/3)
            for i in range(n_path):
                path0 = []
                path_start = content[4+3*i].split()
                path_end = content[4+3*i+1].split()
                path0.append(path_start[3])
                path0.append(path_end[3])
                if path_start[3] not in Kpoint.keys():
                    Kpoint[path_start[3]] = [float(path_start[0]),float(path_start[1]),float(path_start[2])]
                if path_end[3] not in Kpoint.keys():
                    Kpoint[path_end[3]] = [float(path_end[0]),float(path_end[1]),float(path_end[2])]                
                Kpath.append(path0)
            return n_kpoint,Kpoint,Kpath
       
        def generate_k_path(n_kpoint,Kpoint,Kpath):
            kpath = 0
            for path in Kpath:
                start_point = Kpoint[path[0]]
                end_point = Kpoint[path[1]]
                if isinstance(kpath,int):
                    kpath = np.linspace(start_point,end_point,n_kpoint)
                else:
                    kpath = np.concatenate((kpath,np.linspace(start_point,end_point,n_kpoint)),axis=0)
            return kpath
                                
        def get_band(file):
            band = []
            content = file.readlines()
            n_lines = len(content)
            n_point = int(content[1].split()[4])
            n_band = int(content[1].split()[5])
            reading_band = False
            index = 0
            for i in range(n_lines):
                if ' \n' == content[i]:
                    reading_band = False
                    band.append(band0)
                if reading_band:       
                    line = content[i].split()
                    try:                    ### 按有自旋的文件读取
                        band0.append([float(line[0]),float(line[1]),float(line[2])])
                    except:                 ### 按没有自旋的文件读取
                        band0.append([float(line[0]),float(line[1])])
                if 'Band-Index' in content[i]:
                    reading_band = True
                    band0 = [] 
            band.append(band0)
            return n_band,np.array(band)
                        
            
        folder_path = os.path.dirname(self.file_path)
        band_file = open(os.path.join(folder_path,"BAND.dat"),"r")
        Kpoint_file = open(os.path.join(folder_path,"KPOINTS"),"r")
        k_labels_file = open(os.path.join(folder_path,"KLABELS"),"r")
        n_kpoint,Kpoint,Kpath = get_Kpoint(Kpoint_file)
        k_path = generate_k_path(n_kpoint,Kpoint,Kpath)
        n_band,band_data = get_band(band_file)
        for i in range(n_band):
            index_sort = np.argsort(band_data[i][:,0])
            band_data[i] = band_data[i][index_sort]
        self.content["n_kpoint"] = n_kpoint
        self.content["kpath"] = Kpath
        self.content["k_vector"] = k_path
        self.content["energy"] = np.transpose(band_data,(1,0,2))[:,:,1:3]
            
    def __get_npz_content(self):
        """
        储存band的npz文件有两个部分，分别是k_point和energy，二者一一对应，也就是k_point的个数和energy的个数应该相等
        k_point的文件形状是(n_point,3)
        energy的文件形状是(n_point,n_band)
        """
        with np.load(self.file_path) as data:
            self.content["k_vector"] = data["k_vector"]
            self.content["energy"] = data["energy"]
            self.content["kpath"] = data["kpath"].tolist()
            self.content["n_kpoint"] = data["n_kpoint"]
        
        
        
        
if __name__ == "__main__":
    # band_in = BandDataIn("/home/hp/users/kfh/DFTBAI1/example/test_TB/Si_like/Si_PC/BAND.dat")
    band_in = BandDataIn("/home/hp/users/kfh/DFTBAI1/example/BAND-total/Fe-fm/BAND.dat")
    print(band_in.content["k_vector"].shape)
    print(band_in.content['energy'].shape)
    print(band_in.content["energy"][:,4])
    
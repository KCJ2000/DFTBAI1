import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from files_api.Files_out import FilesOut
import numpy as np
import os


class BandDataOut(FilesOut):
    def __init__(self,file_path,content={}) -> None:
        super(BandDataOut,self).__init__()
        self.before_step = FilesOut()
        self.file_path = file_path
        self.description = "存储{}文件，记录band信息".format(self.file_path)
        self.content = content
        self.file_type = os.path.splitext(file_path)[1:][0]
        
        self.check()
        
        
    def check(self):
        if self.file_type == ('',):
            AssertionError("这不是一个文件路径，请输入正确路径")
        
        
    def get_content(self,content:dict):
        self.content = content
        
        
    def save_content(self):
        self.__module_choose()
        
    
    def __module_choose(self):
        if self.content == {}:
            Warning("{}文件输出任务中，并没有储存任何值".format(self.file_path))
        if self.file_type == ".npz":
            self.__save_content_npz()
            
            
    def __save_content_npz(self):
        """
        储存band的npz文件有两个部分，分别是k_point和energy，二者一一对应，也就是k_point的个数和energy的个数应该相等
        k_point的文件形状是(n_point,3)
        energy的文件形状是(n_point,n_band)
        """
        np.savez(self.file_path,**self.content)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from symmetry_operation.sym_operation import SymOp
from symmetry_operation.space_group.sg_operation import SpaceGroupOp
import json
import numpy as np


class MagneticGroupOp(SymOp):
    def __init__(self,group_name=None):
        super(MagneticGroupOp,self).__init__()
        folder_path = os.path.dirname(__file__)
        with open(os.path.join(folder_path,"Magnetic_Group.json")) as f:
            self.group_operation_table = json.load(f)
        self.space_group = SpaceGroupOp()
        
        if group_name!= None:
            self.define_name(group_name)
            
        
    def define_name(self,group_name):
        try:
            ###磁群编号命名
            name = self.group_operation_table.keys()
            if group_name in name:
                self.group_name = self.group_operation_table[group_name]["Name"]
                self.define_operation(group_name)
                sgn = self.group_name["UNI_Number"].split(".")[0]
                self.space_group.define_name(sgn)
            else: raise AssertionError("")
        except:
            ###其他名称命名
            have_name = False
            name = self.group_operation_table.keys()
            for name0 in name:
                Name = self.group_operation_table[name0]["Name"]
                name_type  = Name.keys()
                for name_type0 in name_type:
                    if group_name == Name[name_type0]:
                        self.group_name = Name
                        self.__define_operation(name0)###这里name0是磁群序号
                        sgn = self.group_name["UNI_Number"].split(".")[0]
                        self.space_group.define_name(sgn)
                        have_name = True
                        break 
            if not have_name:
                raise AssertionError("请参照Magnetic_Group.json或者magnetic_table_bns.txt得到需要的磁群命名")
    
            
    def __repr__(self):
        if self.group_name == None:
            return "尚未将该类定义为一个特定的空间群"
        else:
            return "Magnetic Space name:" + str(self.group_name) +"\n" + self.space_group.__repr__()
            
    
    def extra_matrix(self,pg_op:str,trans:list,time_reverse:bool):
        """
            磁群 中spin是要跟着空间转动的，而且转动的矩阵的行列式与是否Time Reverse有关
            如果Time Reverse == true 那么按det为负的旋转矩阵
            如果Time Reverse == false 那么按det为正的旋转矩阵
        """
        matrix = np.zeros((7,7))
        space_matrix_rep = self.space_group.extra_matrix(pg_op,trans)
        rot_matrix_rep = space_matrix_rep[0:3,0:3]
        matrix[0:4,0:4] = space_matrix_rep
        matrix[4,4] = 1
        if time_reverse:
            matrix[4:7,4:7] = -np.matmul(np.eye(3),np.linalg.det(rot_matrix_rep)*rot_matrix_rep)
        else:
            matrix[4:7,4:7] = np.matmul(np.eye(3),np.linalg.det(rot_matrix_rep)*rot_matrix_rep)
        return matrix
            
            
    def __define_operation(self,group_name):
        """
            该函数返回用于作用原子坐标表示上的matrix
        """
        operation_rep = {}
        self.group_operation = self.group_operation_table[group_name]["Operators"]
        if type(self.group_operation) == dict:
            self.group_operation = self.group_operation["BNS"]
        n_op = len(self.group_operation)
        for i in range(n_op):
            operation_rep[str(self.group_operation[i])] = self.extra_matrix(self.group_operation[i][0],
                                                                            self.group_operation[i][1],
                                                                            self.group_operation[i][2])
            
        self.group_operation_rep = operation_rep
            
    
    def elem_mat(self,a:list,b:list):
        """magnetic group 的群乘法操作
        Args:
            a (list): 操作一的list [rot,trans,TR]
            b (list): 同上
        """
        TR = a[2] ^ b[2]### 异或操作
        op = self.space_group.elem_mat(a[0:1],b[0:1])
        return op + [TR]
            
            
    def get_rot_op_rep(self):
        op_rep = {op[0]:self.space_group.pointgroup.jones_faithful_rep[op[0]] for op in self.group_operation}
        return op_rep
        
            
    def basis_trans(self,trans_matrix):
        Trans = np.zeros((7,7))
        Trans[0:3,0:3] = trans_matrix
        Trans[4:7,4:7] = trans_matrix
        Trans[3,3] = 1
        operations = self.group_operation_rep.keys()   
        group_operation_rep_trans = {}     
        for operation in operations:
            group_operation_rep_trans[operation] = Trans@self.group_operation_rep[operation]@np.linalg.inv(Trans)
        return group_operation_rep_trans
        
    
if __name__ == "__main__":
    mg = MagneticGroupOp("227.128")
    trans_matrix = np.linalg.inv(np.array([[0,1/2,1/2],[1/2,0,1/2],[1/2,1/2,0]]))
    print(mg.group_operation)
    # print(mg.group_operation_rep)
    print(mg)
    mg.basis_trans(trans_matrix)
    print(mg.get_rot_op_rep())
    # print(mg.group_operation_rep["['C4x+', [0.25, 0.0, 0.25], False]"])
    # print(mg.group_operation_rep)
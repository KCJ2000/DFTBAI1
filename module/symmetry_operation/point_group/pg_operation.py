import numpy as np
import sympy as sp
from sympy import Matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from symmetry_operation.sym_operation import SymOp
import json

class PointGroupOp(SymOp):
    def __init__(self,group_name=None):
        super(PointGroupOp,self).__init__()
        # self.op_type = None
        # if not( op_type=="cubic" or op_type=="hex" ):
        #     raise AssertionError("PointGroupOp中只有cubic和hex两种类型")
        self.operation_table = {
            "cubic":["E","C2x","C2y","C2z","C31+","C32+","C33+","C34+",
             "C31-","C32-","C33-","C34-","C4x+","C4y+","C4z+","C4x-",
             "C4y-","C4z-","C2a","C2b","C2c","C2d","C2e","C2f",
             "I","Sigma_x","Sigma_y","Sigma_z","S61-","S62-","S63-","S64-",
             "S61+","S62+","S63+","S64+","S4x-","S4y-","S4z-","S4x+",
             "S4y+","S4z+","Sigma_da","Sigma_db","Sigma_dc","Sigma_dd","Sigma_de","Sigma_df"],
            "hex":["E","C6+","C3+","C2","C3-","C6-","C21p","C22p",
             "C23p","C21pp","C22pp","C23pp","I","S3-","S6-","Sigma_h",
             "S6+","S3+","Sigma_d1","Sigma_d2","Sigma_d3","Sigma_v1","Sigma_v2","Sigma_v3"]
        }
        self.x,self.y,self.z = sp.symbols('x'),sp.symbols('y'),sp.symbols('z')
        ###xyz_operation这里用到的是Jones表示，需要和用了Jones表示的Bravais lattice一起使用
        ###但是对于群元之间的运算而言是一样的
        folder_path = os.path.dirname(__file__)
        with open(os.path.join(folder_path,"xyz_operation.json")) as f:
            self.xyz_operation = json.load(f)
        self.xyz_operation = {k:Matrix(v) for k,v in self.xyz_operation.items()}
        
        self.point_operation_type = self.xyz_operation.keys()
        self.jones_faithful_rep = {operation:self.extra_matrix(operation) for operation in self.point_operation_type}
        
        with open(os.path.join(folder_path,"Point_Group.json")) as f:
            self.group_operation_table = json.load(f)
        if group_name != None:   
            self.define_name(group_name)
        
    def define_name(self,group_name):
        if group_name in self.group_operation_table.keys():
            self.group_name = group_name
            self.group_operation = self.group_operation_table[self.group_name]
            self.group_operation_rep = self.get_operation_rep()
        else:
            AssertionError("没有{}群，请从以下群名中选一个{}".format(group_name,self.group_operation_table.keys()))
    
    
    def __repr__(self):
        if self.group_name == None:
            return "尚未将该类定义为一个特定的点群"
        else:
            return "Point Group name:"+self.group_name
    
    
    def extra_matrix(self,operation_name:str):
        """extra_matrix from xyz_operation.json文件

        Args:
            operation_name (str): 操作名称

        Returns:
            np.arrya: 3*3 Rotation Matrix
        """
        xyz_list = self.xyz_operation[operation_name]
        rot_matrix = np.zeros((3,3))
        orgin_xyz = [self.x,self.y,self.z]
        for i in range(3):
            for j in range(3):
                rot_matrix[i][j] = xyz_list[i].coeff(orgin_xyz[j])
        return rot_matrix
        
        
    def elem_mat(self,a:str,b:str):
        mat_result = np.matmul(self.jones_faithful_rep[a],self.jones_faithful_rep[b])
        op_result = next((k for k, v in self.jones_faithful_rep.items() if np.array_equal(v,mat_result)), None)
        return op_result
    

    def get_operation_rep(self):
        operation_rep = {}
        n_op = len(self.group_operation)
        for i in range(n_op):
            operation_rep[self.group_operation[i]] = self.extra_matrix(self.group_operation[i])
        return operation_rep
        
        
    def get_rot_op_rep(self):
        return self.group_operation_rep
    
    
    def basis_trans(self,trans_matrix):
        operations = self.group_operation_rep.keys()
        group_operation_rep_trans = {}
        for operation in operations:
            group_operation_rep_trans[operation] = trans_matrix@self.group_operation_rep[operation]@np.linalg.inv(trans_matrix)
        return group_operation_rep_trans


if __name__ == '__main__':
    # op_type = "Oh"
    # print(op_type)
    pg = PointGroupOp()
    rot_mat = pg.extra_matrix("C3-")
    print(rot_mat)
    result_ele = pg.elem_mat("C2x","C4z+")
    print(result_ele)
    print(pg.group_operation_rep)
    pg = PointGroupOp("C6v")
    print(pg.group_operation_rep)
    
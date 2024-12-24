import numpy as np
import sympy as sp
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from symmetry_operation.sym_operation import SymOp
from symmetry_operation.point_group.pg_operation import PointGroupOp


class SpaceGroupOp(SymOp):
    def __init__(self,group_name=None):
        super(SpaceGroupOp,self).__init__()
        folder_path = os.path.dirname(__file__)
        with open(os.path.join(folder_path,"Space_Group.json")) as f:
            self.group_operation_table = json.load(f)
        with open(os.path.join(folder_path,"sgn2Point.json")) as f:
            self.sgn2Point_table = json.load(f)
        with open(os.path.join(folder_path,"sgn2Schoenflies.json")) as f:
            self.sgn2Schoenflies_table = json.load(f)
            
        self.pointgroup = PointGroupOp()
        self.sgn = None
        if group_name!= None:
            self.define_name(group_name)
        
        
    def define_name(self,group_name):
        try:group_name = int(group_name)
        except:pass
        if isinstance(group_name,int) and (group_name>230 or group_name<1):
            raise ValueError("只有230个空间群，现在输入{}，请重新输入1-230之间的整数".format(group_name))
        elif isinstance(group_name,int):
            self.group_name = self.group_operation_table[str(group_name)]["Name"]
            point_name = self.sgn2Point_table[str(group_name)]
            self.pointgroup.define_name(point_name)
            self.__define_operation(group_name) 
        elif isinstance(group_name,str):
            have_found = False
            for k,v in self.sgn2Schoenflies_table.items():
               if group_name == v:
                    self.group_name = self.group_operation_table[str(k)]["Name"]
                    point_name = self.sgn2Point_table[str(k)]
                    self.pointgroup.define_name(point_name)
                    self.__define_operation(k)
                    have_found = True
                    break
            if not have_found:
                raise ValueError("你输入的空间群名称{}错误，请重新输入，从以下名称中选择{}".format(group_name,self.sgn2Schoenflies_table.values()))

    def __repr__(self):
        if self.group_name == None:
            return "尚未将该类定义为一个特定的空间群"
        else:
            return "Space Group name:" + self.group_name["UNI_Symbol"] +"\n"+ self.pointgroup.__repr__()
       
    
    def extra_matrix(self,pg_op:str,trans:list):
        """得到E(3)群元的表示

        Args:
            pg_op (str):point group种操作的名称
            trans (list): 平移操作1*3list,代表xyz的平移
        Returns:
            np.array:4*4的矩阵，代表E(3)群元的矩阵的表示
        """
        rotmat = self.pointgroup.extra_matrix(pg_op)
        trans = np.array(trans).transpose()
        E_3_rep = np.zeros((4,4))
        E_3_rep[0:3,0:3] = rotmat
        E_3_rep[0:3,3] = trans
        E_3_rep[3,3] = 1
        return E_3_rep
     
    
    def elem_mat(self,a:list,b:list):
        """Space Group中对称性操作的群元乘法
            Space Group由点群中的旋转操作加上平移操作组成
            公式：[R1|T1][R2|T2] = [R1R2|R1T2+T1]
        Args:
            a (二元list): 操作1的点群操作和操作1的平移操作
            b (二元list): 操作2的点群操作和操作2的平移操作
        """
        pg_result = self.pointgroup.elem_mat(a[0],b[0])
        trans_result = np.matmul(self.pointgroup.extra_matrix(a[0]),np.array(b[1]).transpose())+np.array(a[1]).transpose()
        trans_result = trans_result % 1
        trans_result = list(trans_result)
        return [pg_result,trans_result]
        
        
    def __define_operation(self,name):
        """name是空间群序号"""
        self.group_operation = self.group_operation_table[str(name)]["Operators"]
        n_op = len(self.group_operation)
        operation_rep = {}
        for i in range(n_op):
            operation_rep[str(self.group_operation[i])] = self.extra_matrix(self.group_operation[i][0],
                                                                            self.group_operation[i][1])
        self.group_operation_rep = operation_rep


    def get_rot_op_rep(self):
        op_rep = {str(op):self.pointgroup.jones_faithful_rep[op[0]] for op in self.group_operation}
        return op_rep
    
    def basis_trans(self,trans_matrix):
        Trans = np.zeros((4,4))
        Trans[0:3,0:3] = trans_matrix
        Trans[3,3] = 1
        operations = self.group_operation_rep.keys()     
        group_operation_rep_trans = {}   
        for operation in operations:
            group_operation_rep_trans[operation] = Trans@self.group_operation_rep[operation]@np.linalg.inv(Trans)
        return group_operation_rep_trans
    
    
if __name__ == '__main__':
    space = SpaceGroupOp(155)
    print(space.group_name)        
    print("space_group_operation",space.group_operation)
    print("space_group_operation_rep",space.group_operation_rep)
    print(space.pointgroup.group_name)
    print(space)
    a=["C32+",[0,0,0]]
    b=["C33+",[0,0,0]]
    mat_result = space.elem_mat(a,b)
    print(mat_result)
    print(space.extra_matrix("Sigma_x",[0.75,0.75,0.75]))
    
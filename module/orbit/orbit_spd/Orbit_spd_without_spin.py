import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
from orbit.orbit_spd.Orbit_spd import orbit_spd

import sympy as sp
import numpy as np
from numpy import exp,cos,sin
from sympy import simplify,Matrix,symbols,total_degree,expand,parse_expr
import json



class orbit_spd_without_spin(orbit_spd):
    def __init__(self,orbit_list=None):
        super().__init__()
        """这里所有轨道都不需要研究spin,所有的对称性操作不需要考虑SU2群的操作
            只考虑spd轨道的旋转矩阵
        Args:
            orbit_list (list, optional): 用于记录一个wyckoff position上的一套spd轨道 Defaults to None.
            self.orbit_table 是从文件中读取的spd轨道信息
        """ 
            
        self.orbit_type = "spd_without_spin"
        self.check_input(orbit_list)
        self.orbit_num_dict = self.check_orbit_num(orbit_list)
        self.n_orbit = len(orbit_list)

        self.orbit_list = orbit_list
        self.order_dict = self.__check_order()
        self.have_completed_list = [] ###用于记录针对某一个操作是否补全轨道
                
        self.basis_dict = super().init_basis()
        self.trans_matrix_dict,self.orbit_trans_dict = super().trans_basis_matrix()
    
        
    def check_input(self,orbit_list):
        """检查input格式是否符合规范

        Args:
            orbit_list (list): 输入的轨道列表
        """
        if not orbit_list:
            raise AssertionError("忘记输入orbit轨道种类了,请输入")
        
        orbit_name = self.orbit_table.keys()
        for orbit in orbit_list:
            if not orbit in orbit_name:
                raise AssertionError("orbit中不允许设定oribt_rep_table.json外的轨道，请从以下轨道中重新选择{}，如果要新的轨道，请重新设定json文件".format(orbit_name))
    
    
    def check_orbit_num(self,orbit_list):
        orbit_num_dict = {}
        for orbit in orbit_list:
            if orbit in orbit_num_dict.keys():
                orbit_num_dict[orbit] += 1
            else:
                orbit_num_dict[orbit] = 1
        return orbit_num_dict
    
    
    def __check_order(self):
        orbit_name = self.orbit_table.keys()
        order_dict = {}
        for orbit0 in orbit_name:
            self.orbit_table[orbit0] = expand(simplify(self.orbit_table[orbit0],local_dict={'x': self.x, 'y': self.y, 'z':self.z}))### 这个locals是为了保持这个类里面的相同id，但是好像没什么用
            order_dict[orbit0] = int(total_degree(self.orbit_table[orbit0]))
            
            
        return order_dict
            
                      
    def rotation_basis_trans(self,Jones_rot_matrix_dict,latt_vector):
        """
            该函数的作用是将输入进来的Jones表示下的rot_matrix_dict中所有的matrix全都用latt换成实空间正交坐标系下的matrix
            latt.T*rot*inv(latt.T)
        """     
        rot_matrices_name = Jones_rot_matrix_dict.keys()    
        latt = np.transpose(latt_vector)
        inv_latt = np.linalg.inv(latt)
        for rot_matrix_name in rot_matrices_name:
            rotation_matrix = Jones_rot_matrix_dict[rot_matrix_name]
            orthometric_rotation = np.matmul(latt,rotation_matrix)
            orthometric_rotation = np.matmul(orthometric_rotation,inv_latt)
            Jones_rot_matrix_dict[rot_matrix_name] = orthometric_rotation
        return Jones_rot_matrix_dict        
       
    
    def get_orbit_rotation_matrix(self,rotation_matrix):
        """该函数用于把输入的在实空间以正交坐标系为基底的rotation matrix作用于orbit的basis
            得到orbit的rotation matrix

        Args:
            rotation_matrix (np.array): 以正交坐标系为基底的旋转矩阵
        """
        ###定义用于sub的新的符号基，因为sympy中sub的bug，如果sub dict中的如果替换后的变量和之后要替换的变量重合,那么变量替换的前后顺序会影响结果
        ### 比如：x*y subs({x:y+x,y:-x}) 会先替换x,(y+x)*y;然后替换y，(-x+x)*y。
        ### 所以需要定义新的符号变量用于替换,最后再替换回来
        rx,ry,rz = symbols("rx ry rz")
        r_symbol_basis = sp.Matrix([rx,ry,rz])
        roted_orbit_basis = rotation_matrix*r_symbol_basis
        
        need_order = set([self.order_dict[orbit] for orbit in self.orbit_list])
        orbit_rot_dict = {}
        for order in need_order:
            basis_list = Matrix(self.basis_dict[order])
            n_term = len(basis_list)
            basis_list_trans = Matrix(self.basis_dict[order])
            basis_list_trans = basis_list_trans.subs({self.x:roted_orbit_basis[0],
                                                    self.y:roted_orbit_basis[1],
                                                    self.z:roted_orbit_basis[2]})
            basis_list_trans = Matrix([expand(item)for item in basis_list_trans])
            basis_list_trans = basis_list_trans.subs({rx:self.x,ry:self.y,rz:self.z})
            matrix = np.zeros((n_term,n_term))
            for i in range(n_term):
                for j in range(n_term):
                    matrix[i,j] = basis_list_trans[i].coeff(basis_list[j])
            matrix = np.matmul(self.trans_matrix_dict[order],matrix)
            matrix = np.matmul(matrix,np.linalg.pinv(self.trans_matrix_dict[order]))###两个矩阵维数不同，做基变换的时候要做伪逆
            orbit_rot_dict[order] = np.where(abs(matrix)<1e-9,0,matrix)
            
        return orbit_rot_dict                    
           
    
    def complete_basis(self,orbit_rot_dict,rot_name):
        """用户输入的basis可能不完备，我们需要基于orbit_rot_dict和self.orbit_trans_dict中的信息，对self.orbit_list中的轨道进行自动补全
        注意轨道个数之间的关系，如果有相互关联的轨道，所有关联轨道的个数应该规定为其中有最高个数的轨道的个数

        Args:
            orbit_rot_dict (_type_): 我们通过同一个阶的一套完备基得到了一个基于这套完备基的rotation matrix
                            也就是输入的orbit_rot_dict中的内容，完备基的顺序储存在self.orbit_trans_dict中
            rot_name (str): 已经完成对称性测试的旋转操作名称
                            !!!一定要确保名称和得到orbit_rot_dict的操作对应
        """
        ### 检查是否检查过该对称性的完备性
        if rot_name in self.have_completed_list:
            return 0
        ### order_orbit_dict是self.orbit_list中的orbit按各自的阶数进行分类
        order_orbit_dict = {}
        for orbit in self.orbit_list:
            order = self.order_dict[orbit]
            if order in order_orbit_dict.keys():
                if orbit not in order_orbit_dict[order]:
                    order_orbit_dict[order].append(orbit)
            else:
                order_orbit_dict[order] = [orbit]
                
        # print("order_orbit_dict",order_orbit_dict)
        order_orbit_index_dict = {key:{self.orbit_trans_dict[key].index(orbit):self.orbit_num_dict[orbit] for orbit in value} for key, value in order_orbit_dict.items() }
        # print("order_orbit_index_dict",order_orbit_index_dict)
        need_order = set(order_orbit_dict.keys())
    
        ### 检查在给定的orbit_rot_matrix条件(给定的rotation matrix,详情见get_orbit_rotation_matrix函数)下，基是否完备
        complete_basis = []
        for order in need_order:
            basis_index_dict = order_orbit_index_dict[order]
            matrix = orbit_rot_dict[order]
            for index0 in list(basis_index_dict.keys()):
                related_index = np.where(matrix[index0,:] != 0)[0]### 返回一个二阶array，我们只需要一阶的
                if index0 not in related_index:
                    related_index = np.append(related_index,index0) 
                max_num = max([basis_index_dict[related_index0] for related_index0 in related_index if related_index0 in basis_index_dict.keys()])
                for related_index0 in related_index:
                    if related_index0 not in basis_index_dict.keys():### 检查旋转矩阵中是否出现了原本设定中没有的轨道，并进行补全
                        print("rot_name  {},第{}阶，补全轨道{},轨道个数{}".format(rot_name,order,self.orbit_trans_dict[order][related_index0],max_num))
                        basis_index_dict[related_index0] = max_num
                    if basis_index_dict[related_index0] < max_num:
                        print("rot_name  {},第{}阶，轨道{}补全轨道个数{}->{}".format(rot_name,order,self.orbit_trans_dict[order][related_index0],basis_index_dict[related_index0],max_num))
                        basis_index_dict[related_index0] = max_num
                    
                # print(order,"   ",related_index)
            for i in basis_index_dict.keys():### 得到完备的轨道基
                complete_basis = complete_basis + [self.orbit_trans_dict[order][i]]*basis_index_dict[i]
        
        related_dict = {}
        for order in need_order:
            basis_index_dict = order_orbit_index_dict[order]
            matrix = orbit_rot_dict[order]
            for index0 in basis_index_dict.keys():
                related_index = np.where(matrix[index0,:] != 0)[0]
                if index0 not in related_index:
                    related_index = np.append(related_index,index0) 
                if order in related_dict.keys():
                    related_dict[order].append([self.orbit_trans_dict[order][related_index0] for related_index0 in related_index])
                else:
                    related_dict[order] = [[self.orbit_trans_dict[order][related_index0] for related_index0 in related_index]]
        
        ### 去掉在该对称性下重复的部分
        def remove_duplicates(two_d_list):
            seen = set()
            result = []
            for lst in two_d_list:
                # 使用tuple来表示列表，因为列表不能作为集合的元素
                tuple_version = tuple(lst)
                if tuple_version not in seen:
                    seen.add(tuple_version)
                    result.append(lst)
            return result
        for order in  need_order:
            related_dict[order] = remove_duplicates(related_dict[order])
        
        self.orbit_list = complete_basis
        self.orbit_num_dict = self.check_orbit_num(self.orbit_list)
        self.n_orbit = len(self.orbit_list)
        self.have_completed_list.append(rot_name)
        return related_dict
        
     
    def get_orbit_rot_dict(self,rot_op_dict,latt_vector,group_operation_list):
        """
        对一整个群的群操作进行轨道分析,
        群在symmetry_operation文件夹下定义过了这里请给出point_group的三维旋转操作
        

        Args:
            rot_op_dict (dict): 定义的点群中所有操作的名称和matrix 表示(Jones表示下),可以用群类中的get_rot_op函数得到
            latt (np.array): 晶格矢量
            group_operation_list: 磁群的操作列表,有磁群的get_operation函数得到
        
        return group_orbit_rot_matrix_dict (dict):每个operation的旋转操作名称和是否Time Reverse 对应orbit基组的rotation matrix
        """
        
        if len(group_operation_list[0]) == 3:### 在该类中，group_operation_list没什么用，加上1是为了函数形式上的统一，2是加判断条件，以免造成人为编程上的差错
            raise AssertionError("输入的可能是Magnetic Group的操作，orbit_spd_without_spin类只负责Space Group的操作，如有需要，请使用orbit_spd_with_spin类")
        rot_op_dict = self.rotation_basis_trans(rot_op_dict,latt_vector)    
    
        ### 完成所有的群旋转操作
        rot_op_name = rot_op_dict.keys()
        group_orbit_rot_dict = {} ### 用于记录群中所有的旋转操作所对应的该阶所有orbit的旋转矩阵
        group_orbit_rot_matrix_dict = {} ###(函数返回值)用于记录我们输入的轨道在某个操作下所对应的rotation matrix
        for key0 in rot_op_name:
            orthometric_rotation = rot_op_dict[key0]
            ### 得到不同阶的basis的旋转矩阵
            orbit_rot_dict = self.get_orbit_rotation_matrix(orthometric_rotation)
            # print("orbit_rot_dict",orbit_rot_dict)
            group_orbit_rot_dict[key0] = orbit_rot_dict
            ### 利用每个basis的旋转矩阵，使基完备
            self.complete_basis(orbit_rot_dict,key0)                  
        self.matrix_dim = self.n_orbit
        
        ### 完成对于完备化后的orbit_list轨道的分块旋转矩阵构造
        order_orbit_index_dict = {}###用于不同order分块矩阵的构造的index
        orbit_matrix_index = {}### 用于最后把所有的orbit放入最终得到的matrix的index
        matrix_index = 0
        orbit_name = self.orbit_num_dict.keys()
        for orbit in orbit_name:
            index = self.orbit_trans_dict[self.order_dict[orbit]].index(orbit)
            if self.order_dict[orbit] in order_orbit_index_dict.keys():
                order_orbit_index_dict[self.order_dict[orbit]].append(index)
            else:
                order_orbit_index_dict[self.order_dict[orbit]] = [index]
            orbit_matrix_index[orbit] = [matrix_index,matrix_index+self.orbit_num_dict[orbit]]
            matrix_index = matrix_index + self.orbit_num_dict[orbit]
        self.orbit_matrix_index = orbit_matrix_index
        # print("orbit_matrix_index",orbit_matrix_index)
        # print("order_orbit_index_dict",order_orbit_index_dict)
        
        ### 对每一个operation构建矩阵，储存在group_orbit_rot_matrix_dict中（函数返回值）
        for op in rot_op_name:
            orbit_rot_matrix = np.zeros((self.matrix_dim,self.matrix_dim),dtype=complex)
            orbit_rot_dict = group_orbit_rot_dict[op]
            need_order = set(orbit_rot_dict.keys())
            for order in need_order:
                index = order_orbit_index_dict[order]
                n = len(index)
                block_matrix = orbit_rot_dict[order]
                
                ### 如果是有相同轨道个数的话，有两种情况，1是通过对称性操作相互关联的轨道，2是恰好相等的。
                ### 第1种：我们可以将其用单位矩阵进行直积，这样得到的block所描述的轨道关系是分套的，不同套的轨道之间不相互影响。比如有两套px,py轨道，但是其中的每个轨道只与自己对应的那一套轨道反应，对另一套不产生影响
                ### 第2种：旋转轨道时，无关轨道，矩阵元操作应当为0，不影响.所以不处理，if后面不需要接else
                for i in range(n):
                    for j in range(n):
                        index1 = orbit_matrix_index[self.orbit_trans_dict[order][index[i]]]
                        index2 = orbit_matrix_index[self.orbit_trans_dict[order][index[j]]]
                        orbit_num1 = index1[1] - index1[0]
                        orbit_num2 = index2[1] - index2[0]
                        if orbit_num1 == orbit_num2:
                            orbit_rot_matrix[index1[0]:index1[1],index2[0]:index2[1]] = np.kron(block_matrix[index[i],index[j]],np.eye(orbit_num1))
            group_orbit_rot_matrix_dict[op] = orbit_rot_matrix
        return group_orbit_rot_matrix_dict
    
    def __len__(self): 
        """共有多少个轨道，也就是所对应的matrix维数有多少"""
        return self.matrix_dim
    
    
import time
from symmetry_operation.space_group.sg_operation import SpaceGroupOp
from symmetry_operation.mag_group.mg_operation import MagneticGroupOp

if __name__ == "__main__":
    start_time = time.time()
    file_path_out = "G:\\DFTBAI\\DFAITB1\\develop_test\\test_orbit\\orbit_spd_without_spin.txt"
    f = open(file_path_out,"w")
    
    test_group = SpaceGroupOp("191")
    rot_table = test_group.get_rot_op_rep()
    op_list = test_group.group_operation
    
    # test_group = MagneticGroupOp("191.234")
    # rot_table = test_group.get_rot_op_rep()
    # op_list = test_group.group_operation
    
    latt = np.array([[1,0,0],[-1/2,3**0.5/2,0],[0,0,3]])
    
    physics_orbit = orbit_spd_without_spin(["s","s","px","px","pz"])
    group_rot_matrix = physics_orbit.get_orbit_rot_dict(rot_table,latt,op_list)
    print("order_dict",physics_orbit.order_dict)
    print("trans_matrix_dict",physics_orbit.trans_matrix_dict)
    print("orbit_trans_dict",physics_orbit.orbit_trans_dict)
    print(physics_orbit.orbit_list)
    print(physics_orbit.orbit_num_dict)
    f.write(str(group_rot_matrix))
    # print(rot_table)
    # print(op_list)
    # print(len(op_list[0]))
    # print(physics_orbit.trans_matrix_dict)
    # print(physics_orbit.orbit_trans_dict)
    # print(physics_orbit.orbit_num_dict)
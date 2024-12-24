import os
import sys
file_path = sys.argv[0]
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
from orbit.Orbit import orbit

import sympy as sp
import numpy as np
from numpy import exp,cos,sin
from sympy import simplify,Matrix,symbols,total_degree,expand,parse_expr
from sympy.core.evalf import EvalfMixin
import json
from scipy.spatial.transform import Rotation


class orbit_spd(orbit):
    def __init__(self,orbit_list=None,spin_dict=None) -> None:
        """
            orbit_list:用于记录一个wyckoff position上的一套轨道
            spin_dict:用于记录对应的轨道上面要不要加自旋
            self.orbit_table 是从文件中读取的轨道信息，
            self.__check_order()得到self.orbit_table中所有轨道的阶数
        """
        self.orbit_type = "spd"
                
        folder_path = os.path.dirname(__file__)
        with open(os.path.join(folder_path,"orbit_rep_table.json")) as f:
            self.orbit_table = json.load(f) 
    
        self.x,self.y,self.z = symbols("x y z",commutative = True)
        self.xyz_basis = sp.Matrix([self.x,self.y,self.z])
        
        
        
    
    def check_input(self):
        pass
    
    
    def __check_order(self):
        pass
    
    
    def init_basis(self):
        basis_dict = {0:[1]}
        max_order = max(self.order_dict.values())
        basis_pre = [self.x,self.y,self.z]
        basis_new = []
        for order in range(1,max_order+1):
            if order -1 :
                for symbol0 in self.xyz_basis:
                    for basis_pre0 in basis_pre:
                        basis_new.append(basis_pre0*symbol0)
                basis_new = set(basis_new)
                basis_new = list(basis_new)
                basis_pre = basis_new
                basis_new = []
            basis_dict[order] = basis_pre
        return basis_dict
    
    
    def trans_basis_matrix(self):
        """
            在python中，从一个多项式中提取另一个多项式的系数不是很容易，所以我们采取先把所有的项分成单个的(没有加法)
            然后把所有的单个项分成orbit所需的线性组合，也就是Matrix，完成单项和orbit多项式的basis转化，也就是所谓的“trans”
            所以，matrix是根据orbit的order来的也就是传统的spd
            matrix.shape :((2*order+1)，(order+1)*(order+2)/2) 
                        前面是轨道个数，,轨道顺序记录在orbit_trans_dict中，可查
                        后面是单项的个数,单项的顺序储存在self.basis_dict中，可查
            return (dict)   orbit_name:matrix
        """
        trans_matrix_dict = {}
        orbit_trans_dict = {}
        order_key = self.basis_dict.keys()
        
        for order in order_key:
            order_orbit_name = [key for key in self.order_dict if self.order_dict[key] == order]
            orbit_trans_dict[order] = order_orbit_name
            num_orbit = 0
            matrix = np.zeros((int(2*order+1),int((order+1)*(order+2)/2)),dtype=float)
            for orbit in order_orbit_name:
                num_term = 0
                for term in self.basis_dict[order]:
                    matrix[num_orbit,num_term] = self.orbit_table[orbit].coeff(term)
                    num_term += 1
                num_orbit += 1
            trans_matrix_dict[order] = matrix
        return trans_matrix_dict, orbit_trans_dict
 
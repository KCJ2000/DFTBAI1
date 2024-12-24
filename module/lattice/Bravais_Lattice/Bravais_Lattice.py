import numpy as np
import json
import sympy as sp
from sympy import Matrix,cos,sin,csc,cot,pi
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from lattice.Lattice import lattice

class bravaislattice(lattice):
    def __init__(self, lattice_type=None,bravais_lattice_type=None
                 ) -> None:
        super(bravaislattice,self).__init__(lattice_type)
        if bravais_lattice_type == None or bravais_lattice_type not in sum(list(self.crystal_lattice_table.values()),[]):###这里sum用于合并列表
            raise AssertionError("请输入正确的Bravais Lattice名称   {}".format(self.crystal_lattice_table.values()))
        self.bravais_lattice_type = bravais_lattice_type ###这里也可以用Decoration Pattern的思路，但是整体结构简单，这么写也可以
        self.lattice_type = next((k for k, v in self.crystal_lattice_table.items() if self.bravais_lattice_type in v), None)
        folder_path = os.path.dirname(__file__)
        with open(os.path.join(folder_path,"Bravais_Lattice_Vector.json")) as f:
            self.lattice_vector_table = json.load(f)
        with open(os.path.join(folder_path,"Bravais_Lattice_Jones.json")) as f:
            self.Conven2Prim_table = json.load(f)
        self.lattice_vector = Matrix(self.lattice_vector_table[self.bravais_lattice_type])
        self.Conven2Prim_matrix = np.array(self.Conven2Prim_table[self.bravais_lattice_type])
        self.lattice_volumn = self.mixed_product(self.lattice_vector[0,:],self.lattice_vector[1,:],self.lattice_vector[2,:])
        self.repi_lattice_vector = Matrix(np.zeros((3,3)))
        self.repi_lattice_vector_derive()
        
     
    def vector_cross(self,vector_a,vector_b):####向量叉乘
        return Matrix([vector_a[1]*vector_b[2]-vector_a[2]*vector_b[1],
                vector_a[2]*vector_b[0]-vector_a[0]*vector_b[2],
                vector_a[0]*vector_b[1]-vector_a[1]*vector_b[0]]).transpose() 
    def mixed_product(self,vector_a,vector_b,vector_c):###用混合积求体积
        return vector_a[0]*vector_b[1]*vector_c[2]+vector_a[1]*vector_b[2]*vector_c[0]+vector_a[2]*vector_b[0]*vector_c[1]-vector_a[2]*vector_b[1]*vector_c[0]-vector_a[0]*vector_b[2]*vector_c[1]-vector_a[1]*vector_b[0]*vector_c[2]
    def repi_lattice_vector_derive(self):###求该晶格基矢量组，所对应的倒空间基矢量组
        # print(self.lattice_vector[0,:],self.lattice_vector[1,:],self.lattice_vector[2,:])
        self.repi_lattice_vector[0,:] = 2*pi/self.lattice_volumn*self.vector_cross(self.lattice_vector[1,:],self.lattice_vector[2,:])
        self.repi_lattice_vector[1,:] = 2*pi/self.lattice_volumn*self.vector_cross(self.lattice_vector[2,:],self.lattice_vector[0,:])
        self.repi_lattice_vector[2,:] = 2*pi/self.lattice_volumn*self.vector_cross(self.lattice_vector[0,:],self.lattice_vector[1,:])
        
        
    def __repr__(self):
        return "lattice type:" + self.lattice_type + "\n" + "bravais type:" + self.bravais_lattice_type + "\n" +"lattice_vector:\n"+ str(self.lattice_vector) + "\n"
        
        
if __name__ == '__main__':
    bl = bravaislattice(bravais_lattice_type="HexaPrim")
    print(bl.lattice_vector.subs({"a":1,"c":3}))
    print(bl.bravais_lattice_type)
    print(bl.lattice_type)
    print(bl.lattice_volumn)
    print(bl.repi_lattice_vector)
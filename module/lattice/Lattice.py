import numpy as np
import json
import sympy as sp
from sympy import Matrix,cos,sin,csc,cot
import os

class lattice:
    def __init__(self,lattice_type:str) -> None:
        """
        这个之后做无规体系？需要加点东西
        """
        folder_path = os.path.dirname(__file__)
        with open(os.path.join(folder_path,"Crystal2Bravais.json")) as f:
            self.crystal_lattice_table = json.load(f)
        if not(lattice_type in self.crystal_lattice_table.keys() or lattice_type == None):
            raise AssertionError("Please choose one of the Crystall Lattice {}".format(self.crystal_lattice_table.keys()))
        self.lattice_type = lattice_type
        
        
        
        
if __name__ == '__main__':
    lat = lattice("Cubic")
    print(lat.lattice_type)
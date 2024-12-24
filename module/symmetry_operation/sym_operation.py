import numpy as np
import sympy as sp

class SymOp:
    def __init__(self) -> None:
        self.group_name = None
        self.group_operation_table = None
        self.group_operation = None
        self.group_operation_rep = None
        
    def define_name(self):
        pass
    def extra_matrix(self):
        pass
    def elem_mat(self):
        pass
    def __repr__(self):
        pass
    def get_operation_rep(self):
        pass
    def basis_trans(self):
        pass
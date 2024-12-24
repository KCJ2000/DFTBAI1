import numpy as np

class Hamiltonian:
    def __init__(self,dim=1) -> None:
        self.dim = dim
        self.__have_generated_hamiltonian = False
        self.name = "Hamiltonian"
        
    def __generator(self):
        """针对某一种Hamiltonian模型，用于产生self.matrix的函数
        """
        pass
    
    
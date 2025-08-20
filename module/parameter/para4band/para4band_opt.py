import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from parameter.para4band.para4band import Para4Band

"""
    该文件是针对确定了参数，但是
"""

class Para4Band_Optimization(ParaTB_train):
    def __init__(self,model_path,mask_index = None,zero_index = None, device = None):
        super(Para4Band_train,self).__init__(model_path,mask_index,zero_index,device)
        self.para4TB = Para4Band()




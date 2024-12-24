import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from files_api.Files_in4txt import FilesInTxt
import json

class magnetic_table_FileIn(FilesInTxt):
    def __init__(self):
        super(magnetic_table_FileIn,self).__init__()
        self.file_path = "G:\\DFTBAI\\DFAITB1\\symmetry_operation\\mag_group\\magnetic_table_bns.txt"
        self.description = "读取magnetic_table_bns.txt,并得到所有的关于磁群的数据,content中的数据是关于1651个磁群的字典"
        self.getContent()
    
    def open_file(self):
        f = open(self.file_path,"r")
        return f
    
    def getContent(self):
        f = self.open_file()
        begin_read = False
        begin_read_group = False
        have_in_number = 0
        magnetic_group_info = []
        s=0
        for line in f:
            if line == "Magnetic Space Groups\n":
                s+=1
                if s == 2:### 第二个Magnetic Space Groups才开始读，第一个是文件第一行
                    print("开始读取磁群信息")
                    begin_read = True
            
            if begin_read:
                if line == "----------\n" and begin_read_group == False:
                    begin_read_group = True
                elif line!="----------\n" and begin_read_group == True:
                    magnetic_group_info.append(line)
                elif line=="----------\n" and begin_read_group == True:
                    # print("magnetic_group_information:\n",
                    #       magnetic_group_info)
                    have,maggroup = self.getMagGroupInfo(magnetic_group_info)
                    have_in_number += have
                    #print("have_in_number:\n",have_in_number)
                    # print("maggroup:\n",maggroup)
                    self.content[have_in_number] = maggroup
                    magnetic_group_info = []
                
    def getMagGroupInfo(self,lines):
        have=1
        maggroup = {"Name":{
                        "UNI_Number":None,
                        "UNI_Symbol":None,
                        "BNS_Number":None,
                        "BNS_Symbol":None,
                        "OG_Number":None,
                        "OG_Symbol":None},
                    "Operators":None,
                    "Wyckoff":None    
                    }
        
        reading_name = False
        reading_operators = False
        reading_operators_BNS = False
        reading_operators_OG = False
        reading_Wyckoff = False
        reading_Wyckoff_BNS = False
        reading_Wyckoff_OG = False
        
        equal_posi_name = ""
        for line in lines:
            line = line.split(":")
            if line[0] == "UNI":
                reading_name = True
                UNI_name = line[1].split()
                maggroup["Name"]["UNI_Number"] = UNI_name[0]
                maggroup["Name"]["UNI_Symbol"] = UNI_name[1]
                BNS_name = line[2].split()
                maggroup["Name"]["BNS_Number"] = BNS_name[0]
                maggroup["Name"]["BNS_Symbol"] = BNS_name[1]
                OG_name = line[3].split()
                maggroup["Name"]["OG_Number"] = OG_name[0]
                maggroup["Name"]["OG_Symbol"] = OG_name[1]
                reading_name = False
                
                
            if "Operators" in line[0]:
                if "BNS" in line[0]:
                    operators = {}
                    operators["BNS"] = line[1].split()
                    reading_name = False
                    reading_operators = False
                    reading_operators_BNS = True
                    reading_operators_OG = False
                    reading_Wyckoff =False
                    reading_Wyckoff_BNS = False
                    reading_Wyckoff_OG = False
                    continue   
                elif "OG" in line[0]:
                    operators["OG"] = line[1].split()
                    reading_name = False
                    reading_operators = False
                    reading_operators_BNS = False
                    reading_operators_OG = True
                    reading_Wyckoff =False
                    reading_Wyckoff_BNS = False
                    reading_Wyckoff_OG = False
                    continue
                else:
                    operators = line[1].split()
                    reading_name = False
                    reading_operators = True
                    reading_operators_BNS = False
                    reading_operators_OG = False
                    reading_Wyckoff =False
                    reading_Wyckoff_BNS = False
                    reading_Wyckoff_OG = False
                    continue
                
            if "Wyckoff positions" in line[0]:
                if "BNS" in line[0]:
                    Wyckoff = {}
                    if len(line) == 1:
                        Wyckoff["BNS"] = {"lattice_trans":None}
                    else: Wyckoff["BNS"] = {"lattice_trans":line[1].split()}
                    Wyckoff["BNS"]["positions"] = {} 
                    reading_name = False
                    reading_operators = False
                    reading_operators_BNS = False
                    reading_operators_OG = False
                    reading_Wyckoff =False
                    reading_Wyckoff_BNS = True
                    reading_Wyckoff_OG = False
                    continue
                elif "OG" in line[0]:
                    if len(line) == 1:
                        Wyckoff["OG"] = {"lattice_trans":None}
                    else: Wyckoff["OG"] = {"lattice_trans":line[1].split()}
                    Wyckoff["OG"]["positions"] = {} 
                    reading_name = False
                    reading_operators = False
                    reading_operators_BNS = False
                    reading_operators_OG = False
                    reading_Wyckoff =False
                    reading_Wyckoff_BNS = False
                    reading_Wyckoff_OG = True
                    continue
                else:
                    Wyckoff = {}
                    if len(line) == 1:
                        Wyckoff["lattice_trans"] = None
                    else:Wyckoff["lattice_trans"] = line[1].split()
                    Wyckoff["positions"] = {} 
                    reading_name = False
                    reading_operators = False
                    reading_operators_BNS = False
                    reading_operators_OG = False
                    reading_Wyckoff = True
                    reading_Wyckoff_BNS = False
                    reading_Wyckoff_OG = False
                    continue
        
            if reading_operators:
                operators = operators + line[0].split()
            
            if reading_operators_BNS:
                operators["BNS"] = operators["BNS"] + line[0].split()
                
            if reading_operators_OG:
                operators["OG"] = operators["OG"] + line[0].split()
                
            ### 这里try和except是因为有的时候lattice_trans一行可能写不完
            ### 但是如果按照以上使用bool变量的方式也可以，但是我懒得改了（24.06.06）
            if reading_Wyckoff:
                line = line[0].split()
                try:
                    if self.judge_equal_positions(line[0]):
                        Wyckoff["positions"][line[0]] = line[1:]
                        equal_posi_name = line[0]
                    else:
                        Wyckoff["positions"][equal_posi_name] = Wyckoff["positions"][equal_posi_name] + line
                except:
                    Wyckoff["lattice_trans"] = Wyckoff["lattice_trans"] + line
                   
            if reading_Wyckoff_BNS:
                line = line[0].split()
                try:
                    if self.judge_equal_positions(line[0]):
                        Wyckoff["BNS"]["positions"][line[0]] = line[1:]
                        equal_posi_name = line[0]
                    else:
                        Wyckoff["BNS"]["positions"][equal_posi_name] = Wyckoff["BNS"]["positions"][equal_posi_name] + line
                except:
                    Wyckoff["BNS"]["lattice_trans"] = Wyckoff["BNS"]["lattice_trans"] + line

            if reading_Wyckoff_OG:
                line = line[0].split()
                try:
                    if self.judge_equal_positions(line[0]):
                        Wyckoff["OG"]["positions"][line[0]] = line[1:]
                        equal_posi_name = line[0]
                    else:
                        Wyckoff["OG"]["positions"][equal_posi_name] = Wyckoff["OG"]["positions"][equal_posi_name] + line
                except:
                    Wyckoff["OG"]["lattice_trans"] = Wyckoff["OG"]["lattice_trans"] + line
        
        
        maggroup["Operators"] = operators
        maggroup["Wyckoff"] = Wyckoff
        
        return have,maggroup
        
    def judge_equal_positions(self,positype):
        if positype[0] == "()":
            return False
        try:
            int(positype[0])
            return True
        except:return False
    
    def Livtin2Schoenflies(self):
        ###把operations里面的旋转符号改成Schoenflies形式
        key = self.content.keys()
        for key0 in key:
            if type(self.content[key0]["Operators"]) == list:
                n_op = len(self.content[key0]["Operators"])
                for i in range(n_op):
                    self.content[key0]["Operators"][i] = self.Livtin2Schoenflies_single(self.content[key0]["Name"]["UNI_Number"],
                                                                                        self.content[key0]["Operators"][i])
            elif type(self.content[key0]["Operators"]) == dict:
                n_op = len(self.content[key0]["Operators"]["BNS"])
                for i in range(n_op):
                    self.content[key0]["Operators"]["BNS"][i] = self.Livtin2Schoenflies_single(self.content[key0]["Name"]["UNI_Number"],
                                                                                               self.content[key0]["Operators"]["BNS"][i])
                n_op = len(self.content[key0]["Operators"]["OG"])
                for i in range(n_op):
                    self.content[key0]["Operators"]["OG"][i] = self.Livtin2Schoenflies_single(self.content[key0]["Name"]["UNI_Number"],
                                                                                              self.content[key0]["Operators"]["OG"][i])

                
    def Livtin2Schoenflies_single(self,uni_number,op:str):
        """把单个操作中的livtin约定符号改成schoenflies的约定符号
        这里的时间反演对称性是通过'来表示的，我们将其换成True and False"""
        op_r = op.split("|")[0]
        op_r = op_r.split("(")[1]
        number = int(uni_number.split(".")[0])
        with open("G:\\DFTBAI\\DFAITB1\\symmetry_operation\\Bravais2sgno.json") as f:
            if_Hex_table = json.load(f)
        with open("G:\\DFTBAI\\DFAITB1\\symmetry_operation\\mag_group\\L2S_xyz.json") as f:
            name_table = json.load(f)
        bravais = next((k for k,v in if_Hex_table.items() if number in v),None)
        if bravais == "HexaPrim" or bravais == "TrigPrim":
            op_r = name_table["hex+"+op_r]
        else: 
            op_r = name_table[op_r]
        # op = "("+op_r+"|"+op.split("|")[1]
        trans = op.split("|")[1]
        trans =  trans.split(")")[0]
        trans = trans.split(",")
        for i in range(3):
            try: trans[i] = float(trans[i])
            except:
                up,down = map(int,trans[i].split("/"))
                trans[i] = up/down
        time_reverse = "'"in op
        op = [op_r,trans,time_reverse]
        return op
    
    
    
if __name__ == "__main__":
    read = magnetic_table_FileIn()
    read.Livtin2Schoenflies()
    with open("G:\\DFTBAI\\DFAITB1\\symmetry_operation\\mag_group\\Magnetic_Group.json","w") as f:
        json.dump(read.content,f,ensure_ascii=False, indent=4)
    
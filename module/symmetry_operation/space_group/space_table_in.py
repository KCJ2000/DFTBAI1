import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import requests
from bs4 import BeautifulSoup
from files_api.Files_in4txt import FilesInTxt
import json

class space_table_FileIn(FilesInTxt):
    def __init__(self,file_path):
        super(space_table_FileIn,self,).__init__(file_path)
        self.header = {'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0"}
        self.description = "从网页https://stokes.byu.edu/iso/isospacegroup.php得到space group信息,所读取的内容是param.txt中的输入信息"
        self.url = "https://stokes.byu.edu/iso/isospacegroupform.php"
        self.param = self.getContent()
        self.space_table = {}
        
    def open_web(self):
        response = requests.post(url=self.url,data=self.param,headers=self.header)
        if response.status_code == 200:
            return response.text
        else:
            print('请求失败，状态码:', response.status_code)
            exit() 
        
    def getContent(self):
        f = open(self.file_path,"r")
        line = f.readline()
        param = {}
        line = line.split("&")
        num_label = len(line)
        for i in range(num_label):
            para = line[i].split("=")
            param[para[0]]=para[1]
        return param
        
    def content_analysis(self,content):
        with open("G:\\DFTBAI\\DFAITB1\\DFTBAI\\symmetry_operation\\point_group\\xyz_operation.json") as f:
            xyz_table = json.load(f)
        lines = content.split("\n")
        name_table = {}
        for line in lines:
            if "Space group:" in line:
                line = line.split()
                name_table["UNI_Number"] = line[2]
                name_table["UNI_Symbol"] = (line[3]+line[4]).replace("<br>","")
            elif "Non-lattice operators:" in line:
                line = line.replace("<b>Non-lattice operators:</b>","")
                line = line.split(";")
                line[-1] = line[-1].replace("<br>","")
                op_list = self.xyz2rotation_name(line,xyz_table)
                # print(line)
        return {"Name":name_table,"Operators":op_list}
    
    def xyz2rotation_name(self,operations:list,xyz_table:dict):
        operation_list = []
        for op in operations:
            op = op.split()[0]
            op = op.split(",")
            op = [op_xyz.replace("(","") for op_xyz in op]
            op = [op_xyz.replace(")","") for op_xyz in op]
            trans = [0,0,0]
            op_xyz = []
            for i in range(3):
                if "/" in op[i]:
                    try:
                        num = op[i].split("+")[-1]
                        op_xyz.append(op[i].replace("+"+num,""))
                        trans[i] = (int(num.split("/")[0])/int(num.split("/")[1]))%1
                    except:
                        num = op[i].split("-")[-1]
                        op_xyz.append(op[i].replace("-"+num,""))
                        trans[i] = (int(num.split("/")[0])/int(num.split("/")[1]))%1
                else:op_xyz.append(op[i])
            op_name = next((k for k,v in xyz_table.items() if v == op_xyz),None)
            operation_list.append([op_name,trans])
        return operation_list
              
    def construct_table(self):
        self.space_group = {}
        for i in range(1,231):
            self.param['sglabel2'] = str(i)
            content = self.open_web()
            self.space_group[str(i)] = self.content_analysis(content)
            print(str(i))
        
if __name__ == "__main__":
    path = os.path.dirname(__file__)
    file_path = os.path.join(path,"para.txt")
    web = space_table_FileIn(file_path)  
    web.construct_table()
    with open("G:\DFTBAI\DFAITB1\DFTBAI\symmetry_operation\space_group\Space_Group.json","w") as f:
        json.dump(web.space_group,f,ensure_ascii=False,indent=4)
    # print(web.param)  
    # web.open_web()
    # # print(web.param)
    


class FilesIn:
    def __init__(self,file_path=""):
        """
        所有读取文件工作流的模板
        self.file_type是某步工作流所处理的文件类型
        self.content(dict):这一步工作流所读取的各种性质，列成一个字典
        self.file_path 读入文件的路径
        self.description 
        """
        self.file_type = None
        self.content = {}
        self.file_path = file_path
        self.description = "开始读入工作流\n"
        self.before_step = None
        
    
    def check(self):
        pass    
    
    def open_file(self):
        pass
        
    def getContent(self):
        self.open_file()
        pass
    
    def describe(self):
        if self.before_step == None:
            return self.description
        elif issubclass(FilesIn):
            return self.before_step.describe() + "  " + self.description
        
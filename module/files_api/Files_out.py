
class FilesOut:
    def __init__(self) -> None:
        self.file_type = None
        self.content = {}
        self.file_path = ""
        self.description = "开始输出工作流\n"
        self.before_step = None
        
        
    def check(self):
        pass
        
    def get_content(self):
        pass
        
    def open_file(self):
        pass
    
    def save_content(self):
        pass    
        
    
    def describe(self):
        if self.before_step == None:
            return self.description
        elif issubclass(FilesOut):
            return self.before_step.describe() + "  " + self.description

from files_api.Files_in import FilesIn


class FilesInTxt(FilesIn):
    def __init__(self,file_path):
        super(FilesInTxt,self).__init__(file_path)
        self.file_type = "txt"
        self.description = "开始读取txt文件"
        self.before_step = FilesIn()
        

        
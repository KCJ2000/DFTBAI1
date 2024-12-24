该文件夹下是管理该程序与各种文件和程序对接的API的
整个文件系统请遵循Decorator Pattern构建原则
    该原则旨在做到
        1、步骤分解，可以让用户在某一个warp后面直接接入自己的工作流程，而不需要了解甚至看一眼该warp之前的操作是如何实现的
        2、对于不同用户为了实现不同功能，它们之间的工作流不一样，针对不一样的地方，不需要对一样的步骤重写，也不会影响其他的工作流
        
Files_in.py中有FilesIn类
    Variants:
        self.file_type 某一步读取文件的类型
        self.constent 一个你需要读取的各种性质的字典
        self.file_path 所要读取的文件路径
        self.description 这一步读取文件操作的具体描述
    Functions:
        self.getContent() 真正读取文件的操作 
        self.open_file() 打开文件，返回地址
        self.describe 使用Decorator Pattern中的经典操作，可以在某一个warp，跟踪工作流


在应用该文件夹下的工具的话，执行具体操作的文件类应该在各自任务的文件夹下
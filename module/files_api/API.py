import subprocess

class API:
    def __init__(self):
        self.software_type = ""
        self.description = ""
        self.cmd = []
        
    def start_process(self,code):
        result = subprocess.run(self.cmd, input=code, text=True, capture_output=True)
        return result.stdout

    def prepare_code(self,args):
        return ""    
    
    def input_adaptor(self,input):
        return input
    
    def output_adaptor(self,output):
        return output
    
    def run(self,input):
        args = self.input_adaptor(input)
        code = self.prepare_code(args)
        result = self.start_process(code)
        result = self.output_adaptor(result)
        return result
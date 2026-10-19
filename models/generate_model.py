import torch 
import torch.nn as nn

class MiniMatrixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128,128,bias=True)
        
    def forward(self,x):
        return self.fc(x)
    
if __name__ == "__main__":
    model = MiniMatrixModel()
    model.eval()
    
    ip = torch.randn(1,128,dtype=torch.float32)
    
    torch.onnx.export(
        model,
        ip,
        "models/llm.onnx",
        export_params = True,
        opset_version=14,
        input_names = ['input'],
        output_names = ['output']
        
    )
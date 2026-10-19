import onnx

model_path = "./llm.onnx"
model = onnx.load(model_path)

for node in model.graph.node:
    if node.op_type in ["MatMul","Gemm"]:
        print(f"Found standard node: {node.op_type}")
        
        node.op_type = "MatMul"
        node.domain = "emulation"
onnx.save(model,"llm_custom.onnx")
    
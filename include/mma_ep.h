#pragma once
#include <onnxruntime_cxx_api.h>
#include <iostream>

struct Kernel{
    Kernel(const OrtApi& api, const OrtKernelInfo* info){}

    void Compute(OrtKernelContext* context);
};
//operator definition
struct Opd : Ort::CustomOpBase<Opd,Kernel>{
    void* CreateKernel(const OrtApi& api,const OrtKernelInfo* info)const {
        return new Kernel(api,info);
    }

    const char* GetName() const { return "MatMul";}

    const char* GetExecutionProviderType() const { return "CPUExecutionProvider";} //to be changed later

    int GetStartVersion() const { return 1; }
    int GetEndVersion() const { return 20; }

    size_t GetInputTypeCount() const {return 3;}
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    
    size_t GetOutputTypeCount() const {return 1;}
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};


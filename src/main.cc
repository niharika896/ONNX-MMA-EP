#include <iostream>
#include <vector>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "mma_ep.h"

int main(){
    try{
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EpTest");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        //register operator
        Opd oper;
        Ort::CustomOpDomain domain("emulation");
        domain.Add(&oper);
        session_options.Add(domain);

        const char* model_path = "../models/llm_custom.onnx";
        Ort::Session session(env,model_path,session_options);

        std::vector<float> input_values(128,1.0f);
        std::vector<int64_t> input_shape = {1,128};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,input_values.data(),input_values.size(),input_shape.data(),input_shape.size());

        //
        Ort::CustomOpDomain custom_domain("emulation");

        const char* input_names[] = {"input"};
        const char* output_names[] = {"output"};

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},input_names, &input_tensor,1,output_names,1
        );
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        for(int i=0;i<5;i++){
            std::cout<<floatarr[i]<<" ";
        }
    }
    catch(const std::exception& e){
        std::cerr << "Error loading model: "<<e.what()<<std::endl;
        return 1;
    }
    return 0;
}
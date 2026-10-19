#include "mma_ep.h"

void Kernel::Compute(OrtKernelContext *context)
{
    Ort::KernelContext ctx(context);

    std::cout << "Intercepted" << std::endl;

    auto input_X = ctx.GetInput(0);
    auto input_W = ctx.GetInput(1);
    auto input_B = ctx.GetInput(2);

    const float *X = input_X.GetTensorData<float>();
    const float *W = input_W.GetTensorData<float>();
    const float *B = input_B.GetTensorData<float>();

    auto X_shape = input_X.GetTensorTypeAndShapeInfo().GetShape();
    auto W_shape = input_W.GetTensorTypeAndShapeInfo().GetShape();

    // X=(M*K) W=(K*N)
    size_t M = X_shape[0];
    size_t K = X_shape[1];
    size_t N = W_shape[1];

    // std::cout << "Matrix X Shape: [" << X_shape[0] << "x" << X_shape[1] << "]" << std::endl;

    // std::vector<int64_t> output_shape = {X_shape[0],W_shape[1]};
    // auto output_tensor = ctx.GetOutput(0,output_shape.data(),output_shape.size());
    // float* output_data = output_tensor.GetTensorMutableData<float>();

    std::vector<int64_t> output_shape = {(int64_t)M, (int64_t)N};
    auto output_tensor = ctx.GetOutput(0, output_shape.data(), output_shape.size());
    float *Y = output_tensor.GetTensorMutableData<float>();

    // X=(M*K) W=(K*N)
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                sum += X[i * K + k] * W[k * N + j];
            }
            Y[i * N + j] = sum + B[j];
        }
    }
}
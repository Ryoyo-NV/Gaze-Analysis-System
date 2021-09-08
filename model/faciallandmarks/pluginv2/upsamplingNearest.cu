#include "cuda_fp16.h"

template<typename T>
__global__ void nearest_neighbor_upsampling( T * __restrict__ odata, const T * __restrict__ idata, const size_t nrElementsIn, const size_t upsamplingFactor, const size_t columnsIn)
{
    // Reference value for Deconvolution layer from TensorRT 4.0 RC (1x32x5x5): 0.2459 ms (TITAN X Pascal) | 0.2175 ms (GTX 1050)
    // Measured value: 0.0036823 ms (TITAN X Pascal, 67x) | 0.0037416 ms (GTX 1050, 58x)

    // Reference value for Deconvolution layer from TensorRT 4.0 RC (1x64x40x40): 0.4772 ms (TITAN X Pascal) | 0.5669 ms (GTX 1050)
    // Measured value: 0.00838435 ms (TITAN X Pascal, 57x) | 0.032248 ms (GTX 1050, 18x)

    // One thread reads in one original value and writes it out to the dedicated output locations
    const int idxIn = blockDim.x * blockIdx.x + threadIdx.x;
    if(idxIn >= nrElementsIn)
        return;
    const int rowIn = idxIn / columnsIn;
    const int colIn = idxIn % columnsIn;
    const int startIdxOut = (rowIn * columnsIn * upsamplingFactor + colIn) * upsamplingFactor;
    const T input_copy = idata[idxIn];
    for(int r = 0; r < upsamplingFactor; r++)
    {
        const int rowIdxOut = startIdxOut + r * columnsIn * upsamplingFactor;
        for(int c = 0; c < upsamplingFactor; c++)
        {
            const int idx_out = rowIdxOut + c;
            odata[idx_out] = input_copy;
        }
    }
}

void deconv_nearest_neighbor_upsampling(cudaStream_t stream, void *d_output, const void *d_input,
    int N, int C, int H, int W, int upsamplingFactor, bool is_fp16)
{
    const int sizeInput = N * C * H * W;
    const int threadsPerBlock = 1024;
    const int blocksInGrid = (sizeInput + threadsPerBlock - 1) / threadsPerBlock;
    if (is_fp16)
    {
        nearest_neighbor_upsampling<half><<<blocksInGrid, threadsPerBlock, 0, stream >>>((half *)d_output, (const half *)d_input, sizeInput, upsamplingFactor, W);
    }
    else
    {
        nearest_neighbor_upsampling<float><<<blocksInGrid, threadsPerBlock, 0, stream >>>((float *)d_output, (const float *)d_input, sizeInput, upsamplingFactor, W);
    }
    cudaGetLastError();
}

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <assert.h>
#include <iostream>
#include <stdio.h>

#define g_H_Fl 80
#define g_W_Fl 80

template <int H, int W> __global__ void softargmax_kernel_half(const __half2 *srcData, 
   __half *dstDataX,
   __half *dstDataY,
   __half *dstDataProb,
   const int HW,
   const int CHW,
   __half beta,
   __half epsilon)
{
    // Shared memory for storing the maxima per row and after reduction per channel (elementary max operation only available for float, fmaxf):
    __shared__ float s_rowMaximum[H];
    __shared__ float s_channelMaximum;
    // Shared memory for storing the intermediate sums per row and per column, and for saving the softmax denominator (as reciprocate, as __half2):
    __shared__ __half s_sumRows[H];
    __shared__ __half s_sumCols[H];
    __shared__ __half2 s_softmaxDenominatorRcp;

    // We will perform all operations except for fmaxf in half2 mode and will hence convert the constant beta to the __half2 data type:
    __half2 betaHalf2 = __half2half2(beta);

    // Instead of fetching input data from global memory twice, we will use a local buffer per thread:
    __half2 dataLocal[W/2];

    // Information we can infer from this thread:
    const int batch = blockIdx.x;
    const int channel = blockIdx.y;
    const int row = threadIdx.x;

    // This is the starting index of the input elements in __half2 strides (hence the division by two)
    const int elemOffset = (batch * CHW + channel * HW + row * W)/2;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PHASE 1: Calculate the maximum per channel for "accurate" Softmax mode.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // 1a) Find maximum per row:

    float maxElem = -FLT_MAX;
    // Using one thread, loop over all columns:
#pragma unroll
    for (int colIdx = 0; colIdx < W/2; ++colIdx)
    {
        // Fetch an src element pair from global memory:
        __half2 elemPair = srcData[elemOffset + colIdx];
        // Save the src element to local buffer:
        dataLocal[colIdx] = elemPair;
        // Take the maximum of both the upper and the lower value:
        float maxElemPair = fmaxf(__high2float(elemPair), __low2float(elemPair));
        // Take the maximum of the previous maximum element and the maximum of the element pair:
        maxElem = fmaxf(maxElem, maxElemPair);
    }

    // Store maximum of this row in shared memory
    s_rowMaximum[row] = maxElem;

    __syncthreads();

    // 1b) Reduce all maxima from previous step to a global channel maximum and store it in the respective variable in shared memory.
    if (row == 0)
    {
#pragma unroll
        for (int rowIdx = 0; rowIdx < H; ++rowIdx)
        {
            maxElem = fmaxf(maxElem, s_rowMaximum[rowIdx]);
        }
        s_channelMaximum = maxElem;
    }
    __half2 tmpSumHalf2 = __half2half2(0.0f);
    __syncthreads();

    

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PHASE 2: Calculate the pixel-wise Softmax numerators in each channel, and compute the output keypoint probability. 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // 2a) Within each channel, calculate the Softmax numerator of each pixel (subtract the maximum from it, multiply with beta, exponentiate).
    //     Store the result in local memory. Also accumulate the result to tmpSumHalf2.

    // Convert the previously found maximum to __half2:
    __half2 maxElemHalf2 = __float2half2_rn(s_channelMaximum);

    // Using one thread per row, loop over all columns:
#pragma unroll
    for (int colIdx = 0; colIdx < W/2; ++colIdx)
    {
        // Take an element pair from the local buffer:
        __half2 elemPair = dataLocal[colIdx];
        
        // Perform the following__half2 operation: f(x) = exp(beta * (x - channelMaximum))
        __half2 result = h2exp( __hmul2(__hsub2(elemPair, maxElemHalf2), betaHalf2) );

        // Accumulate the results (required for the probability results per keypoint/channel):
        tmpSumHalf2 = __hadd2(result, tmpSumHalf2);

        // Overwrite the buffer element with the result since we need it for the final Softargmax step: 
        dataLocal[colIdx] = result;
    }
    s_sumRows[row] = __hadd(__low2half(tmpSumHalf2), __high2half(tmpSumHalf2));
    
    __syncthreads();

    // 2b) Reduce all sums from the previous step to a single sum per channel (acting as the Softmax denominator), apply rcp, and store it in the first element of the shared row array.
    //     Also calculate the output value for channel/keypoint probability by averaging the sum.
    
    if (row == 0)
    {
        // We do not want to divide by 0, that's why we init this value with epsilon:
        __half tmpSumHalf = epsilon;
        __half channelElements = __float2half(float(HW));
        // Using one thread, loop over all rows:
#pragma unroll
        for (int rowIdx = 0; rowIdx < H; ++rowIdx)
        {
            // Accumulate all rows' aggregated values:
            tmpSumHalf = __hadd(tmpSumHalf, s_sumRows[rowIdx]);
        }
        dstDataProb[channel] = __hdiv(tmpSumHalf, channelElements);
        s_softmaxDenominatorRcp = __half2half2(hrcp(tmpSumHalf));
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PHASE 3: Calculate the pixel-wise Softargmax values for each channel/keypoint.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // 3a) Calculate the Softargmax value for each pixel.

    // Use a local copy of the Softmax denominator:
    __half2 softmaxDenominatorRcpHalf2 = s_softmaxDenominatorRcp;
    
    // Convert this thread's row index into __half2:
    __half2 rowHalf2 = __float2half2_rn(float(row));
    // Initialize the row-wise sums for the X and Y outputs (no epsilon needed):
    __half2 sumRowX = __float2half2_rn(0.0f);
    __half2 sumRowY = __float2half2_rn(0.0f);

    // Using one thread per row, loop over all columns:
#pragma unroll
    for (int colIdx = 0; colIdx < W/2; ++colIdx)
    {
        // Grab an element pair from the local buffer, computed from the previous step:
        __half2 elemPair = dataLocal[colIdx];
        // Calculate the pixel-wise Softmax probability of the element pair within this channel:
        __half2 prob =  __hmul2(elemPair, softmaxDenominatorRcpHalf2);
        
        // Accumulate the result of f(prob, col) = prob * col and f(prob, row) = prob * col to the row-wise sums: 
        sumRowX = __hfma2(prob, make_half2(float(2*colIdx), float(2*colIdx+1)), sumRowX);
        sumRowY = __hfma2(prob, rowHalf2, sumRowY);
    }
    // Now sum-reduce the __half2 elements to a single element:
    s_sumRows[row] = __hadd(__low2half(sumRowX), __high2half(sumRowX));
    s_sumCols[row] = __hadd(__low2half(sumRowY), __high2half(sumRowY));
    __syncthreads();

    // 3b) Reduce all sums from the previous step to the output X and Y value for this channel:
    if (row == 0)
    {
        // Init the channel-wise sums for X and Y outputs:
        __half sumChannelX = __float2half(0.0f);
        __half sumChannelY = __float2half(0.0f);
        // Iterate over all sums in shared memory and accumulate them:
#pragma unroll
        for (int rowIdx = 0; rowIdx < H; ++rowIdx)
        {
            sumChannelX += s_sumRows[rowIdx];
            sumChannelY += s_sumCols[rowIdx];
        }
        // Store the outputs in global memory:
        dstDataX[channel] = sumChannelX;
        dstDataY[channel] = sumChannelY;
    }

}

template <int H, int W> __global__ void softargmax_kernel_float(const float *srcData, 
   float *dstDataX,
   float *dstDataY,
   float *dstDataProb,
   const int HW,
   const int CHW,
   float beta,
   float epsilon)
{

    // Shared memory for storing the maxima per row and after reduction per channel:
    __shared__ float s_rowMaximum[H];
    __shared__ float s_channelMaximum;
    // Shared memory for storing the intermediate sums per row and per column, and for saving the softmax denominator (as reciprocate):
    __shared__ float s_sumRows[H];
    __shared__ float s_sumCols[H];
    __shared__ float s_softmaxDenominatorRcp;

    // Instead of fetching input data from global memory twice, we will use a local buffer per thread:
    float dataLocal[W];

    // Information we can infer from this thread:
    const int batch = blockIdx.x;
    const int channel = blockIdx.y;
    const int row = threadIdx.x;

    // This is the starting index of the input elements in float strides
    const int elemOffset = (batch * CHW + channel * HW + row * W);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PHASE 1: Calculate the maximum per channel for "accurate" Softmax mode.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1a) Find maximum per row:
    float maxElem = -FLT_MAX;
#pragma unroll
    for (int colIdx = 0; colIdx < W; ++colIdx)
    {
        // Fetch an src element pair from global memory:
        float elem = srcData[elemOffset + colIdx];
        // Save the src element to local buffer:
        dataLocal[colIdx] = elem;
        // Take the maximum of the previous maximum element and the current element:
        maxElem = fmaxf(maxElem, elem);

    }

    // Store maximum of this row in shared memory
    s_rowMaximum[row] = maxElem;

    __syncthreads();

    // 1b) Reduce all maxima from previous step to a global channel maximum and store it in the respective variable in shared memory.
    if (row == 0)
    {
#pragma unroll
        for (int rowIdx = 0; rowIdx < H; ++rowIdx)
        {
            maxElem = fmaxf(maxElem, s_rowMaximum[rowIdx]);
        }
        s_channelMaximum = maxElem;
    }
    float tmpSum = 0.0f;
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PHASE 2: Calculate the pixel-wise Softmax numerators in each channel, and compute the output keypoint probability. 
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 2a) Within each channel, calculate the Softmax numerator of each pixel (subtract the maximum from it, multiply with beta, exponentiate).
    //     Store the result in local memory. Also accumulate the result to tmpSum.
    maxElem = s_channelMaximum;
     // Using one thread per row, loop over all columns:
#pragma unroll
    for (int colIdx = 0; colIdx < W; ++colIdx)
    {
        // Take an element from the local buffer:
        float elem = dataLocal[colIdx];
        // Perform the following operation: f(x) = exp(beta * (x - channelMaximum))
        float result = expf((elem - maxElem) * beta) ;
        // Accumulate the results (required for the probability results per keypoint/channel):
        tmpSum += result;
        // Overwrite the buffer element with the result since we need it for the final Softargmax step: 
        dataLocal[colIdx] = result;

    }
    s_sumRows[row] = tmpSum;
    
    __syncthreads();

    // 2b) Reduce all sums from the previous step to a single sum per channel (acting as the Softmax denominator), apply rcp, and store it in the first element of the shared row array.
    //     Also calculate the output value for channel/keypoint probability by averaging the sum.

    if (row == 0)
    {
        // We do not want to divide by 0, that's why we init this value with epsilon:
        tmpSum = epsilon;
        // Using one thread, loop over all rows:
#pragma unroll
        for (int rowIdx = 0; rowIdx < H; ++rowIdx)
        {
            // Accumulate all rows' aggregated values:
            tmpSum += s_sumRows[rowIdx];
        }
        dstDataProb[channel] = tmpSum/float(HW);
        s_softmaxDenominatorRcp = 1.0f/tmpSum;

    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PHASE 3: Calculate the pixel-wise Softargmax values for each channel/keypoint.
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 3a) Calculate the Softargmax value for each pixel.

    // Use a local copy of the Softmax denominator:
    float softmaxDenominatorRcp = s_softmaxDenominatorRcp;

    // Initialize the row-wise sums for the X and Y outputs (no epsilon needed):
    float sumRowsX = 0.0f;
    float sumRowsY = 0.0f;
    float rowFloat = float(row);
    for (int colIdx = 0; colIdx < W; ++colIdx)
    {
        // Grab an element pair from the local buffer, computed from the previous step:
        float elem = dataLocal[colIdx];
        float prob =  elem * softmaxDenominatorRcp;

        // Calculate the pixel-wise Softmax probability of the element pair within this channel:
        sumRowsX = fmaf(prob, float(colIdx), sumRowsX);
        sumRowsY = fmaf(prob, rowFloat, sumRowsY);

    }
    s_sumRows[row] = sumRowsX;
    s_sumCols[row] = sumRowsY;
    __syncthreads();

    // 3b) Reduce all sums from the previous step to the output X and Y value for this channel:

    if (row == 0)
    {
        float sumX = 0.0f;
        float sumY = 0.0f;
         // Iterate over all sums in shared memory and accumulate them:
#pragma unroll
        for (int rowIdx = 0; rowIdx < H; ++rowIdx)
        {
            sumX += s_sumRows[rowIdx];
            sumY += s_sumCols[rowIdx];
        }
        // Store the outputs in global memory:
        dstDataX[channel] = sumX;
        dstDataY[channel] = sumY;
    }

}

void softargmax_kernel_fl(cudaStream_t stream, const void* const* inputs, void **outputs,
    const int N, const int C, const int H, const int W, float beta, float epsilon, bool isfp16)
{    
    if (isfp16) {
        softargmax_kernel_half<g_H_Fl, g_W_Fl><<<dim3(N, C, 1), dim3(W), 0, stream>>>(
            reinterpret_cast<const __half2 *>(inputs[0]), reinterpret_cast<__half *>(outputs[0]),
            reinterpret_cast<__half *>(outputs[0]) + C, reinterpret_cast<__half *>(outputs[1]),
            H * W, C * H * W, __float2half(beta), __float2half(epsilon));
    } else {
        softargmax_kernel_float<g_H_Fl, g_W_Fl><<<dim3(N, C, 1), dim3(W), 0, stream>>>(
            reinterpret_cast<const float *>(inputs[0]), reinterpret_cast<float *>(outputs[0]),
            reinterpret_cast<float *>(outputs[0]) + C, reinterpret_cast<float *>(outputs[1]),
            H * W, C * H * W, beta, epsilon);
    }
}

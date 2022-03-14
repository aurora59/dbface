#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
// #include <vector>

using namespace cv;

// image 800 x 600 x 3 unsigned char
// affine 640 x 640 x 3 unsigned char
// affine 320 x 320 x 12 float

__global__ void normalize_gpu_impl(
    unsigned char* image,     // 来源图像，800 x 600 x 3
    float* affine,             // 目标指针，直接对应网络的输入，也就是12 x 320 x 320
    int width,
    int edge
){
    int position = blockIdx.x * blockDim.x + threadIdx.x; // 线程号

    // 1.边界判断：线程号，会存在比jobs大的情况，这时直接退出
    if(position >= edge) return;
    int dx = position % width;
    int dy = position / width;

    float mean[3] = {0.408, 0.447, 0.47};
    float std[3] = {0.289, 0.274, 0.278};
    
    const int channels = 3;
    unsigned char* image_ptr = image +3*position;
    

    #pragma unroll
    for(int i=0; i < channels; ++i){
        float* affine_ptr = affine + i * edge + position;
        *affine_ptr = (image_ptr[i] / 255.0f - mean[i]) / std[i];
    }
    
}

void normalize_gpu(
    unsigned char* image_device,
    float* data_device,
    int height, int width,
    cudaStream_t stream
){


    int jobs = height * width;
    int threads = 512;
    int blocks = ceil(jobs / (float)threads);
    normalize_gpu_impl<<<blocks, threads, 0, stream>>>(
        image_device,
        data_device,
        width,
        jobs
    );

}


#include "pybind11.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <NvOnnxParser.h>
#include <string>
#include <vector>
#include <numeric>
#include "logging.h"


using namespace std;
namespace py = pybind11;

void decode(float* hm, float* hm_pool,
            float* box, float* landmark,
            float* parray, int height,
            int width, cudaStream_t stream);

void normalize_gpu(
    unsigned char* image_device,
    float* data_device,
    int height, int width,
    cudaStream_t stream
);

float desigmoid(float x){
    return -log(1/x-1);
}

const int NUM_BOX_ELEMENT = 16;      // left, top, right, bottom, confidence, class, keepflag

static Logger gLogger;

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d+1, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}



bool save_to_file(const string& file, const void* data, size_t size){
    FILE* handle = fopen(file.c_str(), "wb");
    if(handle == nullptr)
        return false;
    fwrite(data, 1, size, handle);
    fclose(handle);
    return true;
}


vector<unsigned char> load_from_file(const string& file){
    FILE* handle = fopen(file.c_str(), "rb");
    if(handle == nullptr)
        return {};

    //相对末尾偏移0个字节，就是末尾
    fseek(handle, 0, SEEK_END);
    //获取当前文件指针，得到大小
    long size = ftell(handle);

    fseek(handle, 0, SEEK_SET);
    vector<unsigned char> output;
    if(size>0){
        output.resize(size);
        fread(output.data(), 1, size, handle);
    }

    fclose(handle);
    return output;
}


void build(string file_path, string out_path="model.trtmodel", int batch_size=16){

    cudaSetDevice(0);
    // JLogger logger;
    auto builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, gLogger);
    if(!parser->parseFromFile(file_path.c_str(), 1)){
        printf("Parser failed.\n");
        return;
    }

    int min_h = 1;
    int min_w = 1;
    int height = 640;
    int width = 640;
    int max_h = 2 * height;
    int max_w = 2 * width;

    auto input = network->getInput(0);
    auto dims = input->getDimensions();

    int channel = dims.d[1];
    if(dims.d[2] != -1){
        min_h = dims.d[2];
        height = dims.d[2];
        max_h = dims.d[2];
    }

    if(dims.d[3] != -1){
        min_w = dims.d[3];
        width = dims.d[3];
        max_w = dims.d[3];
    }

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input_1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, channel, min_h, min_w});
    profile->setDimensions("input_1", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{int(batch_size/2+1), channel, height, width});
    profile->setDimensions("input_1", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{batch_size, channel, max_h, max_w});

    auto config = builder->createBuilderConfig();
    config->addOptimizationProfile(profile);

    builder->setMaxBatchSize(batch_size);
    config->setMaxWorkspaceSize(1 << 22);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    
    auto engine = builder->buildEngineWithConfig(*network, *config);
    auto host_memory = engine->serialize();
    // auto host_memory = builder->buildSerializedNetwork(*network, *config);
    save_to_file(out_path, host_memory->data(), host_memory->size());
    printf("Done.\n");
    printf("model_data.size() = %d\n", host_memory->size());

    host_memory->destroy();
    engine->destroy();
    config->destroy();
    network->destroy();
    builder->destroy();
}



class Infer{
public:
    Infer(string file="model.trtmodel"){
        cudaSetDevice(0);
        auto model_data = load_from_file(file);
        if(model_data.empty()){
            printf("Load model failure.\n");
            return;
        }

        initLibNvInferPlugins(&gLogger, "");

        runtime = nvinfer1::createInferRuntime(gLogger);
        printf("model_data.size() = %d\n", model_data.size());

        engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
        context = engine->createExecutionContext();


        stream = nullptr;
        cudaStreamCreate(&stream);
        nbBindings = engine->getNbBindings();

        printf("nbBindings = %d\n", nbBindings);
        maxbatchsize = engine->getMaxBatchSize();
        bufferSize.resize(nbBindings);
        buffers.resize(nbBindings);

        size_t max_image_size = maxbatchsize*640*640*3;
        cudaMalloc(&image_device, max_image_size);
        out_data = (float*)malloc(maxbatchsize*16*1000*4);

        for (int i = 0; i < nbBindings; ++i) {
            int64_t totalSize = maxbatchsize*640*640*3*4;
            bufferSize[i] = totalSize;

            printf("bufferSize[%d] = %d\n", i, bufferSize[i]);
            cudaMalloc(&buffers[i], totalSize);

        }
        printf("infer done\n");
    }


    vector<vector<py::array_t<float>>> inference(vector<py::array_t<unsigned char>>& image_numpy);
    ~Infer(){
        for (int i = 0; i < nbBindings; ++i){
            cudaFree(buffers[i]);
        }
        cudaFree(image_device);
        free(out_data);
        cudaStreamDestroy(stream);
        context->destroy();
        engine->destroy();
        runtime->destroy();
    };

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    cudaStream_t stream;
    int nbBindings;
    std::vector<void*> buffers;
    std::vector<int64_t> bufferSize;
    int maxbatchsize;
    void* image_device;
    float* out_data;
};

vector<vector<py::array_t<float>>> Infer::inference(vector<py::array_t<unsigned char>>& image_numpy){
    // 模型推理

    // 1. 图像加载，并预处理


    //    居中缩放
    // int network_size = 640;
    py::buffer_info buf0 = image_numpy[0].request();
    int batch = image_numpy.size();
    int h = buf0.shape[0];
    int w = buf0.shape[1];
    int height = h, width = w, stride = 32;
    if(h % stride != 0){
        height = (h / stride + 1) * stride;
    }
    if(w % stride != 0){
        width = (w / stride + 1) * stride;
    }

    size_t origin_image_size = 3 * h * w;
    size_t image_size = 3 * height * width;
    for(int i=0; i<batch; i++){
        py::buffer_info buf = image_numpy[i].request();
        cudaMemsetAsync(image_device + i*image_size, 0, image_size, stream);
        cudaMemcpyAsync((unsigned char*)image_device + i*image_size, buf.ptr, origin_image_size, cudaMemcpyHostToDevice, stream);
        normalize_gpu(
            (unsigned char*)image_device + i*image_size,
            (float*)buffers[0] + i*image_size,
            height, width,
            stream
        );
    }
    // for(int i=0; i<batch; i++){
    //     printf("%d\n", i);
    //     normalize_gpu(
    //         (unsigned char*)image_device + i*image_size,
    //         (float*)buffers[0] + i*image_size,
    //         height, width,
    //         stream
    //     );
    // }
    // cudaMemcpyAsync(image_device, buf.ptr, image_size, cudaMemcpyHostToDevice, stream);
    // float ratio = (float)network_size / std::max(height, width);
    // // printf("height = %d, width = %d\n", height, width);
    // // printf("ratio = %f\n", ratio);
    // int network_height = ratio*height+0.5f;
    // int network_width = ratio*width+0.5f;
    // // printf("network_height = %d, network_width = %d\n", network_height, network_width);
    // normalize_gpu(
    //     (unsigned char*)image_device,
    //     (float*)buffers[0],
    //     height, width,
    //     network_height,
    //     network_width,
    //     1.0f/ratio,
    //     stream
    // );

    // printf("normal done\n");

    // py::buffer_info buf = image_numpy.request();
    // int batch_size = buf.shape[0];
    // int channel = buf.shape[1];
    // int network_height = buf.shape[2];
    // int network_width = buf.shape[3];
    // size_t image_byte = channel * network_height * network_width * sizeof(float);
    // // unsigned char* img = (unsigned char*)buf.ptr;
    // // printf("buf.ptr[100] = %d\n", img[1000]);

    

    // printf("198\n");

    // for(int i=0; i<batch; ++i){

    //     cudaMemcpyAsync(buffers[0] + i * image_size, buf.ptr + i * image_size, image_size, cudaMemcpyHostToDevice, stream);

    // }

    context->setBindingDimensions(0, nvinfer1::Dims4{batch, 3, height, width});
    context->allInputDimensionsSpecified();

    int bindings = buffers.size();

    void* buffer_end[bindings];
    for(int j=0; j<bindings; j++){
        buffer_end[j] = buffers[j];
    }

    // cudaStreamSynchronize(stream);


    context->enqueue(batch, buffer_end, stream, nullptr);

    nvinfer1::Dims output_dim;
    size_t buffer_size[4];
    for(int i=0; i<4; i++){
        output_dim = context->getBindingDimensions(i+1);
        buffer_size[i] = volume(output_dim);
        // printf("%d\n", buffer_size[i]);
    }
    size_t max_box_size = 1000 * 16;
    for(int i=0; i<batch; i++){
        cudaMemsetAsync((float*)image_device+i*max_box_size, 0, sizeof(int), stream);
        decode((float*)buffers[1]+i*buffer_size[0], (float*)buffers[2]+i*buffer_size[1],
                (float*)buffers[3]+i*buffer_size[2], (float*)buffers[4]+i*buffer_size[3],
                (float*)image_device+i*max_box_size, output_dim.d[2],
                output_dim.d[3], stream);
        // printf("%d, %d\n", output_dim.d[2],
        //         output_dim.d[3]);
        cudaMemcpyAsync(out_data+i*max_box_size, (float*)image_device+i*max_box_size, max_box_size*4, cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);

    vector<vector<py::array_t<float>>> dst(3);
    vector<float> score, box, landmark;
    py::array_t<float> p_score, p_box, p_landmark;
    for(int i=0; i<batch; i++){
        float* parray = out_data+i*max_box_size;
        int count = (int)*parray;
        int numbox = count;
        // printf("%d\n", count);
        for(int j=0; j<count; j++){
            float* pbox = parray + 1 + j*NUM_BOX_ELEMENT;
            int keepflag = pbox[5];
            if(keepflag == 1){
                score.emplace_back(pbox[4]);
                box.insert(box.end(), pbox, pbox+4);
                landmark.insert(landmark.end(), pbox+6, pbox+16);
            }else{numbox--;}
        }
        p_score = py::array_t<float>({numbox}, score.data());
        p_box = py::array_t<float>({numbox, 4}, box.data());
        p_landmark = py::array_t<float>({numbox, 5, 2}, landmark.data());
        dst[0].emplace_back(p_score);
        dst[1].emplace_back(p_box);
        dst[2].emplace_back(p_landmark);
        score.clear();
        box.clear();
        landmark.clear();
    }
  

    return dst;
}



PYBIND11_MODULE(db_infer, m) {


    m.def("build", &build, "onnx2trt", py::arg("file_path"), py::arg("out_path")="model.trtmodel", py::arg("batch_size")=1);
    
    py::class_<Infer>(m, "Infer")
        .def(py::init<string&>())
        .def("inference", &Infer::inference, "输入opencv读入图片，numpy格式");

}
#include <cuda_runtime.h>
#include <stdio.h>

// using namespace std;

const int NUM_BOX_ELEMENT = 16;      // left, top, right, bottom, confidence, class, keepflag

static __device__ float uexp(float x){
    float gate = 1.0f;
    float base = exp(gate);
    if(abs(x) < gate){
        return x * base;
    }

    if(x > 0){
        return exp(x);
    }
    else{
        return -exp(-x);
    }
}

__global__ void decode_gpu(float* hm, float* hm_pool, float* box, float* landmark, float* parray, int width, float threshold, int edge){
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    if(position >= edge) return;
    if(hm[position] == hm_pool[position] && hm[position] >= threshold){
        int cx, cy, min_face=400;
        float x, y, r, b;
        // float* landmark[10];
        cx = position % width;
        cy = position / width;
        x = *(box + 0 * edge + position);
        y = *(box + 1 * edge + position);
        r = *(box + 2 * edge + position);
        b = *(box + 3 * edge + position);
        x = (cx - x) * 4;
        y = (cy - y) * 4;
        r = (cx + r) * 4;
        b = (cy + b) * 4;
        // printf("%f, %f, %f, %f\n", x, y, r, b);
        if((r-x)*(b-y) > min_face && x>0 && y>0 && r>x){
            // printf("%f, %f, %f, %f\n", x, y, r, b);
            int index = atomicAdd(parray, 1);
            // printf("%d\n", index);
            float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
            *pout_item++ = x;
            *pout_item++ = y;
            *pout_item++ = r;
            *pout_item++ = b;
            *pout_item++ = hm[position];
            *pout_item++ = 1; // 1 = keep, 0 = ignore
            for(int i=0; i<5; i++){
                *pout_item++ = (uexp(*(landmark + i * edge + position)*4) + cx)*4;
                *pout_item++ = (uexp(*(landmark + (i+5) * edge + position)*4) + cy)*4;
            }
        }

    }
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float* bboxes, float threshold){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = (int)*bboxes;
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[5] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

void decode(float* hm, float* hm_pool, float* box, float* landmark, float* parray, int height, int width, cudaStream_t stream){
    int area = height * width;
    int threads = 512;
    int blocks = ceil(area/(float)threads);
    float threshold = 0.5f;
    // printf("%d\n", 114);
    decode_gpu<<<blocks, threads, 0, stream>>>(hm, hm_pool, box, landmark, parray, width, threshold, area);
    float num_box = 0.0f;
    cudaMemcpyAsync(&num_box, parray, 4, cudaMemcpyDeviceToHost, stream);
    // printf("%d\n", (int)num_box);
    if(num_box==0){return;}
    threads = num_box > threads ? threads : num_box;
    blocks = ceil(num_box/(float)threads);
    float nms_threshold = 0.5f;
    // printf("%d\n", 120);
    nms_kernel<<<blocks, threads, 0, stream>>>(parray, nms_threshold);
}
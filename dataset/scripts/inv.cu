#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <stack>
#include <cstdlib>
#include <algorithm>
#include <queue>


using namespace std;


__device__ float dot_product(float* i,float* j){
    return i[0]*j[0] + i[1]*j[1];
}

__device__ float cross_product(float* i,float* j){
    return abs(i[0]*j[1] - i[1]*j[0]);
}

__device__ float vector_norm(float* v){
    return pow(pow(v[0],2)+pow(v[1],2),0.5);
}
__global__ void query_all_gpu(float* q_list_x_gpu,float* q_list_y_gpu,
                            float* i_list_x_gpu,float* i_list_y_gpu,float* e_list_x_gpu,float* e_list_y_gpu, float* distance_gpu,
                        int query_length,int node_length)
 {  
    
     for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < query_length; i += blockDim.x * gridDim.x)
     {
         float min_distance = 9999;
         for (int j=0;j<node_length;j++){
            float distance = 0;
            float iq[2] = {q_list_x_gpu[i]-i_list_x_gpu[j],q_list_y_gpu[i]-i_list_y_gpu[j]}; 
            float qe[2] = {e_list_x_gpu[j]-q_list_x_gpu[i],e_list_y_gpu[j]-q_list_y_gpu[i]}; 
            float ie[2] = {e_list_x_gpu[j]-i_list_x_gpu[j],e_list_y_gpu[j]-i_list_y_gpu[j]}; 
            if ((dot_product(iq,ie)>=0)&(dot_product(qe,ie)>=0)){
                distance = cross_product(ie,iq) / vector_norm(ie); 
            }
            else if ((dot_product(iq,ie)<0)&(dot_product(qe,ie)>=0)){
                distance = vector_norm(iq);
            }
            else{
                distance = vector_norm(qe); 
            }
            if (distance < min_distance){
                min_distance = distance;
            }
            if (min_distance<=1){
                break;
            }
         }
         if (min_distance<=1){
             min_distance = 1;
         }
         distance_gpu[i] = min_distance;
     }
 }
 
 void query_gpu(float *init_x, float *init_y, float* end_x, float* end_y, float*query_list_x,float* query_list_y,
    float * distance_list, int node_length, int query_length)
 {
	
    float* q_list_x_gpu = NULL;
    float* q_list_y_gpu = NULL;
    float *i_list_x_gpu = NULL;
    float *i_list_y_gpu = NULL;
    float *e_list_x_gpu = NULL;
    float *e_list_y_gpu = NULL;
    float *distance_gpu = NULL;
    float *temp_list = NULL;

	cudaMallocHost((float**) &temp_list, sizeof(int) * query_length);

    cudaMalloc((float**) &q_list_x_gpu, sizeof(int) * query_length);
	cudaMalloc((float**) &q_list_y_gpu, sizeof(int) * query_length);
	cudaMalloc((float**) &i_list_x_gpu, sizeof(float) * node_length);
	cudaMalloc((float**) &i_list_y_gpu, sizeof(float) * node_length);
    cudaMalloc((float**) &e_list_x_gpu, sizeof(float) * node_length);
	cudaMalloc((float**) &e_list_y_gpu, sizeof(float) * node_length);
    cudaMalloc((float**) &distance_gpu, sizeof(float) * query_length);
     
     cudaMemcpy((float*) q_list_x_gpu, (float*) query_list_x, sizeof(int) * query_length, cudaMemcpyHostToDevice);
     cudaMemcpy((float*) q_list_y_gpu, (float*) query_list_y, sizeof(int) * query_length, cudaMemcpyHostToDevice);
     cudaMemcpy((float*) i_list_x_gpu, (float*) init_x, sizeof(float) * node_length, cudaMemcpyHostToDevice);
     cudaMemcpy((float*) i_list_y_gpu, (float*) init_y, sizeof(float) * node_length, cudaMemcpyHostToDevice);
     cudaMemcpy((float*) e_list_x_gpu, (float*) end_x, sizeof(float) * node_length, cudaMemcpyHostToDevice);
     cudaMemcpy((float*) e_list_y_gpu, (float*) end_y, sizeof(float) * node_length, cudaMemcpyHostToDevice);
     
     query_all_gpu<<<1024, 1024>>>(q_list_x_gpu,q_list_y_gpu,i_list_x_gpu,i_list_y_gpu,
                                    e_list_x_gpu,e_list_y_gpu,distance_gpu,query_length,node_length);
     
     cudaDeviceSynchronize();
    
     cudaMemcpy(temp_list, distance_gpu, sizeof(float) * query_length, cudaMemcpyDeviceToHost);
     for (int i=0;i<query_length;i++)
     {      
         distance_list[i] = temp_list[i];
        //  printf("%f---\n",distance_list[i]);
     }
     
     cudaFree(q_list_y_gpu);
     cudaFree(q_list_x_gpu);
     cudaFree(i_list_y_gpu);
     cudaFree(i_list_x_gpu);
     cudaFree(e_list_y_gpu);
     cudaFree(e_list_x_gpu);
     cudaFree(distance_gpu);
	 cudaFreeHost(query_list_x);
     cudaFreeHost(query_list_y);
     cudaFreeHost(init_x);
     cudaFreeHost(init_y);
     cudaFreeHost(end_x);
     cudaFreeHost(end_y);
     cudaFreeHost(temp_list);
 }
 
 extern "C" {  
     void python_read_data(float *init_x, float *init_y, float* end_x, float* end_y,float*query_list_x,float* query_list_y,
        float * distance_list, int node_length, int query_length){
        
        // printf("node length: %d, query_length: %d\n",node_length,query_length);
        query_gpu(init_x, init_y, end_x, end_y,query_list_x,query_list_y,
            distance_list,  node_length, query_length);
     }
    
}
/*
 * ECB-mode TEA Encryption (MP1, Fall 2020, CS425/Sonoma State University)
 */

// Modified for CUDA by Cole Nicholson-Rubidoux
// For CS425 Homework 4

// Note: I ran this on my local machine using its own CUDA architecture, if it does not compile for you,
//       please change the value of threads_per_block to be a smaller number, like 512 or 256

#include <stdio.h>
#define __STDC_LIMIT_MACROS /* for UINT32_MAX when compiled with nvcc */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Number of elements in the array to encrypt */
#define NUM_ELTS 100000
#define THREADS_PER_BLOCK 1024

// Macro to check for errors
#define CUDA_CHECK(e) { \
     cudaError_t err = (e); \
     if (err != cudaSuccess) \
    {\
       fprintf(stderr, "CUDA error :%s, line %d, %s : %s\n", __FILE__, __LINE__, #e, cudaGetErrorString(err)) ; \
       exit(EXIT_FAILURE); \
    } \
 } 


/* TEA encryption */
// This type of declaration allows us to call the same function on both the host and the device
__device__ __host__ static void encrypt(uint32_t *data, const uint32_t *key) {
    // This defines the code which runs on the device
#ifdef __CUDA_ARCH__
    int thread_idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    if(thread_idx < NUM_ELTS - 1 /*&& thread_idx % 2 == 0*/) {
        uint32_t v0 = data[thread_idx], v1 = data[thread_idx + 1], sum = 0, i;
        uint32_t delta = 0x9e3779b9;
        uint32_t k0 = key[0], k1 = key[1], k2 = key[2], k3 = key[3];
        for (i = 0; i < 32; i++) {
            sum += delta;
            v0 += ((v1 << 4) + k0) ^ (v1 + sum) ^ ((v1 >> 5) + k1);
            v1 += ((v0 << 4) + k2) ^ (v0 + sum) ^ ((v0 >> 5) + k3);
        }
        data[thread_idx] = v0;
        data[thread_idx + 1] = v1;
    }
#else
    uint32_t v0=data[0], v1=data[1], sum=0, i;
    uint32_t delta=0x9e3779b9;
    uint32_t k0=key[0], k1=key[1], k2=key[2], k3=key[3];
    for (i=0; i < 32; i++) {
        sum += delta;
        v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
        v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
    }
    data[0]=v0; data[1]=v1;
#endif
    // This bit above defines the code which runs on the host
}

__global__ void cudaEncryptWrapper(uint32_t *data, const uint32_t *key){
    encrypt(data, key);
}

/* TEA decryption */
static void decrypt(uint32_t *data, const uint32_t *key) {
	uint32_t v0=data[0], v1=data[1], sum=0xC6EF3720, i;
	uint32_t delta=0x9e3779b9;
	uint32_t k0=key[0], k1=key[1], k2=key[2], k3=key[3];
	for (i=0; i<32; i++) {
		v1 -= ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
		v0 -= ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
		sum -= delta;
	}
	data[0]=v0; data[1]=v1;
}
          
int main(void)
{
	printf("Encrypting %d-element array\n", NUM_ELTS);

	/* Generate a random 128-bit encryption key */
	const uint32_t h_key[4] = { rand()%UINT32_MAX, rand()%UINT32_MAX, rand()%UINT32_MAX, rand()%UINT32_MAX };
	size_t key_size = 4 * sizeof(uint32_t);

	/* Allocate memory for the original data and the encrypted data */
	size_t size = NUM_ELTS * sizeof(uint32_t);
	uint32_t *h_data = (uint32_t *)malloc(size);
	uint32_t *h_encrypted = (uint32_t *)malloc(size);
       // added by SHUBBHI for printing final results
	uint32_t *h_output = (uint32_t *)malloc(size);
	uint32_t *h_decrypted = (uint32_t *)malloc(size);

	// Initialize device variables and memory
	uint32_t *d_encrypted, *d_key;
    size_t threads_per_block = 1024;
    size_t number_of_blocks = (NUM_ELTS + threads_per_block - 1) / threads_per_block;
	CUDA_CHECK(cudaMalloc((void**) &d_encrypted, size));
	CUDA_CHECK(cudaMalloc((void**) &d_key, key_size));

	if (h_data == NULL || h_encrypted == NULL){
		fprintf(stderr, "Unable to allocate host memory\n");
		exit(EXIT_FAILURE);
	}

	/* Generate a long sequence of random numbers to encrypt */
	for (int i = 0; i < NUM_ELTS; ++i){
		h_data[i] = rand() % UINT32_MAX;
	}
	printf("\nOriginal data: %u %u ... %u\n", h_data[0], h_data[1], h_data[NUM_ELTS-1]);

	/* Perform ECB-mode TEA encryption on the CPU */
	/* (The GPU's result can be checked against this later) */
	memcpy(h_encrypted, h_data, size);
	for (int i = 0; i < NUM_ELTS; i += 2) {
		encrypt(&h_encrypted[i], h_key);
	}
	printf("\nEncrypted data: %u %u ... %u\n",h_encrypted[0], h_encrypted[1], h_encrypted[NUM_ELTS-1]);
	// added by SHubbhi Taneja
	memcpy(h_decrypted, h_encrypted, size);
	for (int i = 0; i< NUM_ELTS; i+=2) {
	    decrypt(&h_decrypted[i], h_key);
	}
        
	printf("\nDecrypted data: %u %u ... %u\n",h_decrypted[0], h_decrypted[1], h_decrypted[NUM_ELTS-1]);

	/*
	* Encrypt h_data using a CUDA kernel, then copy the encrypted
	* data back to the CPU and verify that it is the same as
	* what was computed on the CPU (above)
	*/
	CUDA_CHECK(cudaMemcpy(d_encrypted, h_data, size, cudaMemcpyHostToDevice)); // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_key, h_key, key_size, cudaMemcpyHostToDevice)); // Copy key to device
    cudaEncryptWrapper<<<number_of_blocks / 2, threads_per_block>>>(d_encrypted, d_key); // Call parallelized encrypt function
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernels to complete
    CUDA_CHECK(cudaMemcpy(h_output, d_encrypted, size, cudaMemcpyDeviceToHost)); // Copy encryption back to host

    /* verify results from GPU*/
    printf("\n encrypted data from device %u %u ... %u \n" , h_output[0],h_output[1],h_output[NUM_ELTS-1]);
    for (int i=0; i< NUM_ELTS;i++){
        if(h_output[i] == h_encrypted[i]) {
            continue;
        }
        else {
            printf("Not matching for element %d \n" , i);
            printf("on CPU the encrypted value is = %u \t", h_encrypted[i]);
            printf(" while on  GPU, its= %u \n", h_output[i]);
        }
    }

         

    /* free device global memory*/
    /* free host memory*/
    free(h_encrypted);
	free(h_data);
	CUDA_CHECK(cudaFree(d_encrypted));
	CUDA_CHECK(cudaFree(d_key));
        
	//reset the device
	CUDA_CHECK(cudaDeviceReset());

	return 0;
}

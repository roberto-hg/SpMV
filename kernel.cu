#define _CRT_SECURE_NO_DEPRECATE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


//Error handling macro, wrap it around cuda function whenever possible
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		system("pause");
		//exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void ErrorHandlMacro_Example()
{
	//almost all cuda function return a cudaError_t type 
	//which can be parsed to check for error 

	//In this example we will use the macro twice
	//first time to wrap a cudaMalloc function, this one will go without errors
	//second time to wrap a cudaMemcpy function where we purposefully make an error
	//where we move more data (6*sizeof(int)) to the GPU side more than what we 
	//have allocated (5*sizeof(int))

	//after the second call, the program will hault and show an error 
	int *d_value;
	int h_value[6] = { 1, 1, 1, 1, 1, 1 };
	HANDLE_ERROR(cudaMalloc((void**)&d_value, 5 * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(d_value, h_value, 6 * sizeof(int), cudaMemcpyHostToDevice));
}


__global__
void Spmvker(int size, double *value, int *col, int *rowoff, int rownum, double *vec, double *output) {

	int id = threadIdx.x;

	if (id <= rownum) {

		output[id] = 0;

		for (int i = rowoff[id]; i < rowoff[id + 1]; i++) {
			output[id] += value[i] * vec[col[i]];
		}
	}
}


void cootocsr(int *rowoff, int *row, int size) {
	//Converts COO row to CSR row offset.

	rowoff[0] = 0;
	int prev = 0, accu = 1, j = 1;

	for (int i = 1; i < size; i++) {
		
		if (row[i] - row[prev] > 1) {
			for (int k = 0; k < row[i] - row[prev]; k++) {
				rowoff[j++] = accu;
			}
			prev = i;
		}

		else if (row[prev] != row[i]) {
			rowoff[j++] = accu;
			prev = i;
		}
		
		accu += 1;
	}

	rowoff[j] = accu;
}


int main(int argc, char *argv[]) {


	///ErrorHandlMacro_Example();


	//************ 1) Read input files ************//	
	// this will have to allocate the necessary memory on the CPU side (Done)

	FILE *pToMFile = fopen(argv[1], "r"); //Opens matrix file.

	int matsize;
	fscanf(pToMFile, "%d", &matsize); //Sets the matrix size.

	int mintsize = matsize * sizeof(int);
	double mdoublesize = matsize * sizeof(double);

	//Creates row, col, and value arrays.
	int *h_row = (int *)malloc(mintsize);
	int *h_col = (int *)malloc(mintsize);
	double *h_mvalue = (double *)malloc(mdoublesize);

	//Allocates the .txt file data to the created arrays.
	for (int i = 0; i < matsize; i++)
	{
		fscanf(pToMFile, "%d", &h_row[i]);
		fscanf(pToMFile, "%d", &h_col[i]);
		fscanf(pToMFile, "%lf", &h_mvalue[i]);
	}

	fclose(pToMFile);

	FILE *pToVFile = fopen(argv[2], "r"); //Opens vector file.

	int veclen;
	fscanf(pToVFile, "%d", &veclen); //Sets the vector size.

	//Creates vector value array.
	int vecbytes = veclen * sizeof(double);
	double *h_vec = (double *)malloc(vecbytes);

	//Allocates the .txt file data to the created array.
	for (int i = 0; i < veclen; i++) {
		fscanf(pToVFile, "%lf", &h_vec[i]);
	}

	fclose(pToVFile);
	//******************************************//


	//Convert COO to CSR by converting h_row to h_rowoff
	int rownum = h_row[matsize - 1] + 1;
	int rowoffsize = (rownum + 1) * sizeof(int);
	int *h_rowoff = (int *)malloc(rowoffsize);

	cootocsr(h_rowoff, h_row, matsize);


	//************ 2) Allocate memory on GPU *************//			
	// allocate memory on GPU side 
	// move the data from the CPU to GPU

	double *d_mvalue;
	int *d_col;
	int *d_rowoff;
	double *d_vec;
	double *d_output;
	int outputsize = rownum * sizeof(double);
	
	HANDLE_ERROR(cudaMalloc((void **)&d_mvalue, mdoublesize));
	HANDLE_ERROR(cudaMalloc((void **)&d_col, mintsize));
	HANDLE_ERROR(cudaMalloc((void **)&d_rowoff, rowoffsize));
	HANDLE_ERROR(cudaMalloc((void **)&d_vec, vecbytes));
	HANDLE_ERROR(cudaMalloc((void **)&d_output, outputsize));

	HANDLE_ERROR(cudaMemcpy(d_mvalue, h_mvalue, mdoublesize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_col, h_col, mintsize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_rowoff, h_rowoff, rowoffsize, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_vec, h_vec, vecbytes, cudaMemcpyHostToDevice));

	//****************************************************//


	//************ 3) Launch the kernel to do the operation on GPU *************//
	//this will launch the kernel of SpMV
	//the kernel should take one 2D array (input matrix), one 1D array (input vector)
	//and output one 1D array (output vector)
	
	Spmvker<<<1, rownum>>>(matsize, d_mvalue, d_col, d_rowoff, rownum, d_vec, d_output);

	//**************************************************************************//


	//************ 4) Move data to CPU *************//	
	//after computing the output vector on the GPU side, we need to move this vector 
	//to the CPU to further processing
	//to do this, we need to allocate memory on the CPU side 
	//then move the data to CPU

	double *h_output = (double *)malloc(outputsize);

	HANDLE_ERROR(cudaMemcpy(h_output, d_output, outputsize, cudaMemcpyDeviceToHost));

	//**********************************************//


	//************ 5) Compute SpMV on CPU & Compare *************//	
	//compute the SpMV on the CPU (serially), the output will be 
	//stored in temp array 
	//compare the output in this temp array with the output from the 
	//GPU (that we have just moved to CPU in 4)

	double *temp = (double *)malloc(outputsize);

	for (int i = 0; i < rownum; i++) {

		temp[i] = 0;

		for (int j = h_rowoff[i]; j < h_rowoff[i + 1]; j++) {
			temp[i] += h_mvalue[j] * h_vec[h_col[j]];
		}
	}

	if (memcmp(temp, h_output, outputsize))
		printf("\nOutput GPU SpMV vector is NOT the same as CPU SpMV vector\n\n");

	else
		printf("\nOutput GPU SpMV vector is the same as CPU SpMV vector\n\n");


	for (int i = 0; i < rownum; i++) {
		printf("Resultant vector element %d = %lf\n\n", i, h_output[i]);
	}
	//**********************************************************//


	//************ 6) Deallocate memory on CPU and GPU *************//	
	//de-allocate memory from GPU and wrap the de-allocation call with 
	//error handling macro (see and run ErrorHandlMacro_Example())
	//this will indicate if there was any error happened in the kernel 
	//(e.g., tried to access memory not allocated)
	//de-allocate memory on the CPU (it is a good practice to do this
	//even though c++ will de-allocate it automatically) 

	//De-allocate CPU memory:
	free(h_row);
	free(h_rowoff);
	free(h_col);
	free(h_mvalue);
	free(h_vec);
	free(temp);
	free(h_output);

	//Deallocate GPU memory:
	HANDLE_ERROR(cudaFree(d_mvalue));
	HANDLE_ERROR(cudaFree(d_col));
	HANDLE_ERROR(cudaFree(d_rowoff));
	HANDLE_ERROR(cudaFree(d_vec));
	HANDLE_ERROR(cudaFree(d_output));

	//************************************************************//

	return 0;

}

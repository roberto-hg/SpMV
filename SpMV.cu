#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/scan.h>


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
	HANDLE_ERROR(cudaMalloc((void**)&d_value, 5*sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(d_value, h_value, 6*sizeof(int), cudaMemcpyHostToDevice));
}


void Array2D(int *row, int *col, double *value, int size) {
	double *mat2D;
	mat2D = (double *) malloc(size * sizeof(double));

	for (int i = 0; i < matsize; i++) {
		mat2D[row[i]][col[i]] = value[i];
	}

	return 0;
}


double *main(int argc, char *argv[]) {

	
	///ErrorHandlMacro_Example();
	

	//************ 1) Read input files ************//	
	// this will have to allocate the necessary memory on the CPU side (Done)

	FILE *pToMFile = fopen(argv[1], "r"); //Opens matrix file.

	int matsize;
	fscanf(pToMFile, "%d", &matsize); //Sets the matrix size.

	int mintsize = matsize * sizeof(int);
	double mdoublesize = matsize * sizeof(double);

	//Creates row, col, and value arrays.
	int *h_row = (int *) malloc(mintsize);				
	int *h_col = (int *) malloc(mintsize);              
	double *h_mvalue = (double *) malloc(mdoublesize);

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
	double *h_vec = (double *) malloc(veclen * sizeof(double));

	//Allocates the .txt file data to the created array.
	for (int i = 0; i < veclen; i++) {
		fscanf(pToVFile, "%lf", &h_vec[i]);
	}

	fclose(pToVFile);
	//******************************************//
		

	Array2D(h_row, h_col, h_mvalue, matsize);


	//************ 2) Allocate memory on GPU *************//			
	// allocate memory on GPU side 
	// move the data from the CPU to GPU

	double *d_mat;
	double *d_vec;
	double *d_output;
	int rowsize = h_row[matsize - 1];
	int finalsize = rowsize * sizeof(double);

	cudaMalloc((void **)&d_mat, mdoublesize);
	cudaMalloc((void **)&d_vec, mdoublesize); 
	cudaMalloc((void **)&d_output, finalsize);

	cudaMemcpy(d_mat, mat2D, mdoublesize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vec, h_vec, veclen * sizeof(double), cudaMemcpyHostToDevice);


	//****************************************************//
	








	//************ 3) Launch the kernel to do the operation on GPU *************//
	//this will launch the kernel of SpMV
	//the kernel should take one 2D array (input matrix), one 1D array (input vector)
	//and output one 1D array (output vector)
	
	

	__global__ void Spmvker(double *d_mat, double *d_vec, int rowsize) {

		int id = threadIdx.x;

		if (id < rowsize) {

			for (int i = 0; i < ; i++) {

				d_mat[id][i] = d_mat[id][i] * d_vec[id][i];
			}
			
			double accu = 0;
			for (int i = 0; i < ; i++) {
				accu += d_mat[id][i];
			}

			d_output[id] = accu;
		}
			
	}
		

	//**************************************************************************//






	//************ 4) Move data to CPU *************//	
	//after computing the output vector on the GPU side, we need to move this vector 
	//to the CPU to further processing
	//to do this, we need to allocate memory on the CPU side 
	//then move the data to CPU

	double *h_output = (double *) malloc(finalsize);

	cudaMemcpy(d_output, h_output, finalsize, cudaMemcpyDevicetoHost);





	//**********************************************//







	//************ 5) Compute SpMV on CPU & Compare *************//	
	//compute the SpMV on the CPU (serially), the output will be 
	//stored in temp array 
	//compare the output in this temp array with the output from the 
	//GPU (that we have just moved to CPU in 4)


	for (int i = 0; i < rowsize; i++) {

	}


	//**********************************************************//






	//************ 6) Deallocate memory on CPU and GPU *************//	
	//de-allocate memory from GPU and wrap the de-allocation call with 
	//error handling macro (see and run ErrorHandlMacro_Example())
	//this will indicate if there was any error happened in the kernel 
	//(e.g., tried to access memory not allocated)
	//de-allocate memory on the CPU (it is a good practice to do this
	//even though c++ will de-allocate it automatically) 

	free(h_row);
	free(h_col);
	free(h_mvalue);
	free(h_vec);
	free(mat2D);


	//************************************************************//



	//Returns pointer to the resultant SpMV vector.
	
	return h_output;
}

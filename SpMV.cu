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



double * vectormult(double *vector, int *col, int size) {
	//Array that has the vector values corresponding to the matrix elements.

	double *p_vector2 = (double *) malloc(size * sizeof(double));

	for (int i = 0; i < size; i++) {
		p_vector2[i] = vector[col[i]];
	}

	free(vector);
	free(col);

	return p_vector2;
}

int indexer(int *row, int *index, int size) {
	//Creates index array that returns the index of each row (uses defined index array).
	//Also returns the index size.

	int prev = 0, j = 0, k = 0;

	for (int i = 1; i < size; i++) {
			if (row[prev] != row[i]) {
					index[j++] = prev;
			k++;
			prev = i;
		}
	}

	index[j] = prev;
	index[j + 1] = size;
	k += 2;

	return k;
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
	int *p_row = (int *) malloc(mintsize);				
	int *p_col = (int *) malloc(mintsize);              
	double *p_mvalue = (double *) malloc(mdoublesize);

	//Allocates the .txt file data to the created arrays.
	for (int i = 0; i < matsize; i++)
	{
		fscanf(pToMFile, "%d", &p_row[i]);
		fscanf(pToMFile, "%d", &p_col[i]);
		fscanf(pToMFile, "%lf", &p_mvalue[i]);
	}

	fclose(pToMFile);

	FILE *pToVFile = fopen(argv[2], "r"); //Opens vector file.

	int veclen;
	fscanf(pToVFile, "%d", &veclen); //Sets the vector size.

	//Creates vector value array.
	double *p_vec = (double *) malloc(veclen * sizeof(double));

	//Allocates the .txt file data to the created array.
	for (int i = 0; i < veclen; i++) {
		fscanf(pToVFile, "%lf", &p_vec[i]);
	}

	fclose(pToVFile);
	//******************************************//
		







	//************ 2) Allocate memory on GPU *************//			
	// allocate memory on GPU side 
	// move the data from the CPU to GPU



	//****************************************************//
	







	//************ 3) Launch the kernel to do the operation on GPU *************//
	//this will launch the kernel of SpMV
	//the kernel should take one 2D array (input matrix), one 1D array (input vector)
	//and output one 1D array (output vector)
	

	//Creates the memory for multiplication.
	double *p_mult = (double *) malloc(mdoublesize);

	//Calls vectormult for multiplication.
	double * vecmu = vectormult(p_vec, p_col, matsize);

	//Multiplies the matrix elements with the corresponding vector values.
	for (int i = 0; i < matsize; i++) {
		p_mult[i] = p_mvalue[i] * vecmu[i];
	}
		
	free(p_mvalue);
	free(vecmu);

	//Calls indexer to create an index array to use on the segmented scanned array.
	int * index = (int *) malloc(mintsize + sizeof(int));  
	int indexsize = indexer(p_row, index, matsize);

	/* Unused Code			
	double *d_mult; 
	int *d_segrow;

	cudaMalloc((void **) &d_mult, mdoublesize);
	cudaMalloc((void **) &d_segrow, mintsize);

	cudaMemcpy(d_mult, p_mult, mdoublesize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_segrow, keys, mintsize, cudaMemcpyHostToDevice);

	//thrust::equal_to<double> binary_pred;
	//thrust::maximum<double> binary_op;
	*/

	//GPU segmented scan using the row array and the multiplied array.
	thrust::inclusive_scan_by_key(p_row, p_row + matsize, p_mult, p_mult);

	free(p_row);

	//Allocates the memory for the resultant vector from the SpMV.
	double *result = (double *) malloc(mdoublesize);

	printf("\n\nResult of SpMV:\n\n");

	//Fetches the coressponding sums from the segmented scan array and
	//creates the resultant vector from the SpMV. Also prints each element.
	for (int i = 1; i < indexsize; i++) {
		result[i] = p_mult[index[i] - 1];
		printf("Element[%d] = %lf\n\n", i-1, result[i]);
	}
	//**************************************************************************//






	//************ 4) Move data to CPU *************//	
	//after computing the output vector on the GPU side, we need to move this vector 
	//to the CPU to further processing
	//to do this, we need to allocate memory on the CPU side 
	//then move the data to CPU



	//**********************************************//







	//************ 5) Compute SpMV on CPU & Compare *************//	
	//compute the SpMV on the CPU (serially), the output will be 
	//stored in temp array 
	//compare the output in this temp array with the output from the 
	//GPU (that we have just moved to CPU in 4)



	//**********************************************************//






	//************ 6) Deallocate memory on CPU and GPU *************//	
	//de-allocate memory from GPU and wrap the de-allocation call with 
	//error handling macro (see and run ErrorHandlMacro_Example())
	//this will indicate if there was any error happened in the kernel 
	//(e.g., tried to access memory not allocated)
	//de-allocate memory on the CPU (it is a good practice to do this
	//even though c++ will de-allocate it automatically) 



	//************************************************************//



	//Returns pointer to the resultant SpMV vector.
	return result;
}

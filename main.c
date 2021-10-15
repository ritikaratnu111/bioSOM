#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include<unistd.h>
#include<sys/time.h>

#include "defs.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

void checkErr(cl_int err, const char *name);
void checkKernelEnqueue(cl_int err);
void showDeviceInfo(cl_device_id device_id);

float beta = BETA_INIT;

void initMatrix(float *x, int row, int col){
	int i = 0;
	int j = 0;
	for(i = 0; i < row; i = i+1){
		for(j = 0; j < col; j = j+1){
			x[i*col+j] = (float) (rand() & MAXVAL);	
		}
	}
}

double rtclock() {
	struct timezone Tzp;
	struct timeval Tp;
	int stat;
	stat = gettimeofday(&Tp, &Tzp);
	if(stat != 0) printf("Error return from gettimeofday: %d",stat);
	return( Tp.tv_sec + Tp.tv_usec*1.0e-6);
}



int main(int argc, char *argv[])
{
	float *IS; //A matrix
	float *W; //B matrix
	float *Wold; //B matrix
	int Sdim, Ndim, Mdim; //IS[S][M], W[N][M], C[N][M]
	int szIS, szW; //number elements
	int MAX_ITER;


	Sdim = S; 
	Mdim = M;
	Ndim = N;
	MAX_ITER = MAXITER;


	if (argc > 1)
	MAX_ITER = atof(argv[1]);
	if (argc > 2)
	Sdim = atof(argv[2]);
	if (argc > 3)
	Mdim = atof(argv[3]);
	if (argc > 4)
	Ndim = atof(argv[4]);


	printf("=== Sequential BioSOM ===\n");
	printf("\t Max. iteration count: %d\n",MAX_ITER);
	printf("\t Input size: %d\n", Sdim);
	printf("\t Sequence length: %d\n",Mdim);
	printf("\t No of neurons: %d\n",Ndim);
	printf("=========================\n");

//////////////////////////////////////KERNEL DECLARATIONS BEGIN ///////////////////////////////////////////
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	
	cl_platform_id platform_id[2]; 
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	FILE *fp;
	char fileName[] = "./som_kernel.cl";
	char *source_str;
	size_t source_size;
     
	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel file.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(2, platform_id, &ret_num_platforms);
	checkErr(ret,"platform_id");

	printf("No. of platforms detected: %d\n",ret_num_platforms);

	for(int i=0; i<ret_num_platforms; i++){

		//Experience the portability of OpenCL, choose either CPU or GPU as the device.
		//ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
		ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);

		if(ret == CL_SUCCESS){
			break;
		}
	}

	checkErr(ret,"device_id");
	showDeviceInfo(device_id);

	/* Create OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	checkErr(ret,"Context"); 
	/* Create Command Queue */
	//cl_queue_properties proprt[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	//command_queue = clCreateCommandQueueWithProperties(context, device_id, proprt, &ret);
	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
	checkErr(ret,"Command Queue"); 
     
	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	checkErr(ret,"Program"); 
     
	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(ret != CL_SUCCESS){
		size_t len;
		char   buffer[2048];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n",buffer);
		exit(EXIT_FAILURE);
	}    
 
	/* Create OpenCL Kernel */
	kernel = clCreateKernel(program, "training", &ret);	

////////////////////////////////////// KERNEL DECLARATIONS END /////////////////////////////////////////////////////

    
	szIS = Sdim*Mdim;
	szW = Ndim*Mdim;

	srand(1);

	IS = (float *)malloc(szIS*sizeof(float));
	W = (float *)malloc(szW*sizeof(float));	
	Wold = (float *)malloc(szW*sizeof(float));	
	initMatrix(IS, Sdim, Mdim);
	initMatrix(W, Ndim, Mdim);


	//----------------- device buffers --------------------
	cl_mem dev_IS; //memory object for IS matrix 	
	dev_IS = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*szIS, NULL, NULL);	
	cl_mem dev_W; //memory object for W matrix 		
	dev_W = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*szW, NULL, NULL);
	cl_mem dev_eucl_dist; //memory object for eucledian distance from one ipvec 		
	dev_eucl_dist = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*Ndim, NULL, NULL);
	/* Set OpenCL Kernel Parameters */
	ret = clSetKernelArg(kernel, 0, sizeof(int), &Sdim);
	ret = clSetKernelArg(kernel, 1, sizeof(int), &Ndim);
	ret = clSetKernelArg(kernel, 2, sizeof(int), &Mdim);
	ret = clSetKernelArg(kernel, 3, sizeof(float), &beta);

	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_IS); checkErr(ret,"Kernel arg 4");
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &dev_W); checkErr(ret,"Kernel arg 5");
    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &dev_eucl_dist); checkErr(ret,"Kernel arg 6"); 
    ret = clSetKernelArg(kernel, 7, sizeof(float)*Mdim, NULL); checkErr(ret,"Kernel arg 7"); 

	size_t 	global[1]; //global domain size
	size_t	local[1];  //local domain size
    size_t  global_offset[1];
    cl_uint nd = 1;
	global[0] = (size_t) Ndim;
	local[0]  = (size_t) (Ndim/8);
    global_offset[0]=0;
	cl_event event;

	double nanoSeconds = 0;
    
    float error;

    for (int iter = 0; iter < MAX_ITER; iter++ )
	{
		error = 0;
		for (int i = 0; i< szW;i++)
		{
			Wold[i] = W[i];
		}

		//------- Write A and B matrices into device memory ----
		ret = clEnqueueWriteBuffer(command_queue, dev_IS, CL_TRUE, 0, sizeof(float)*szIS, IS, 0, NULL, NULL); 
				checkErr(ret,"Write Buffer IS");	
		ret = clEnqueueWriteBuffer(command_queue, dev_W, CL_TRUE, 0, sizeof(float)*szW, W, 0, NULL, NULL);	
				checkErr(ret,"Write Buffer W");	
		
		ret = clEnqueueNDRangeKernel(command_queue, kernel, nd, global_offset, global, local, 0, NULL, &event);
				checkKernelEnqueue(ret);

		/* Wait for the event object to complete */   
		/* cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) */
		clWaitForEvents(1, &event);

		cl_ulong time_start;
	 	cl_ulong time_end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	 	nanoSeconds += time_end-time_start;

		/* Copy results from the memory buffer */
	 	ret = clEnqueueReadBuffer(command_queue, dev_W, CL_TRUE, 0, sizeof(float) * szW, W, 0, NULL, NULL);

	 	beta = fminf(beta*DECAY, MIN_BETA);

	 	for (int i = 0; i< szW;i++)
		{
			error += abs(Wold[i] = W[i]);
		}

	}

	printf("\nError: %f\n",error);

 	printf("OpenCL Execution time: %f ms\n",nanoSeconds/1000000.0);
 
  	 
  	printf("ALL OK WE ARE DONE \n");





	/////////////////////////////////////////////////////////////////////////////////////////
	free(IS);
  	free(W);
 	free(source_str);
 	// free(dot);
    	/* Finalization */
    	ret = clFlush(command_queue);
    	ret = clFinish(command_queue);
    	ret = clReleaseKernel(kernel);
    	ret = clReleaseProgram(program);
    	ret = clReleaseMemObject(dev_IS);
    	ret = clReleaseMemObject(dev_W);
    	ret = clReleaseCommandQueue(command_queue);
    	ret = clReleaseContext(context);
	return 0;

}

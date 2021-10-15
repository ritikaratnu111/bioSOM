#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

void checkErr(cl_int err, const char *name){
	if(err != CL_SUCCESS) {
		printf("ERROR: %s\n",name);
		exit(EXIT_FAILURE);
	}
}

void checkKernelEnqueue(cl_int err){
  if(err != CL_SUCCESS){
	  switch(err){
		  case CL_INVALID_PROGRAM_EXECUTABLE:
			  printf("Invalid program executable\n"); break;
		  case CL_INVALID_COMMAND_QUEUE:
			  printf("Invalid command queue\n"); break;
		  case CL_INVALID_KERNEL:
			  printf("Invalid kernel\n"); break;
		  case CL_INVALID_CONTEXT:
			  printf("Invalid context\n"); break;
		  case CL_INVALID_KERNEL_ARGS:
			  printf("Invalid kernel args\n"); break;
		  case CL_INVALID_WORK_DIMENSION:
			  printf("Invalid work dimension\n"); break;
		  case CL_INVALID_WORK_GROUP_SIZE:
			  printf("Invalid work group size\n"); break;
		  case CL_OUT_OF_HOST_MEMORY:
			  printf("out of host memory\n"); break;
		  default:
			  printf("UNKNOWN\n"); break;
	  }
  }
}

void showDeviceInfo(cl_device_id device_id)
{
	cl_int ret;
	char *value;
	size_t valuesize;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valuesize);
	checkErr(ret, "DEVICE NAME");
	value = (char*)malloc(valuesize);
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, valuesize, value, NULL);
	printf("Device: %s\n",value);
	free(value);

	ret = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 0, NULL, &valuesize);
	checkErr(ret, "DEVICE VERSION");	
	value = (char*)malloc(valuesize);
	clGetDeviceInfo(device_id, CL_DEVICE_VERSION, valuesize, value, NULL);		
	printf("Hardware version: %s\n",value);
	free(value);
 
	ret = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, NULL, &valuesize);
	checkErr(ret, "DRIVER VERSION");	
	value = (char*)malloc(valuesize);
	clGetDeviceInfo(device_id, CL_DRIVER_VERSION, valuesize, value, NULL);		
	printf("Driver version: %s\n",value);
	free(value);

	ret = clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valuesize);
	checkErr(ret, "OpenCL C version");	
	value = (char*)malloc(valuesize);
	clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, valuesize, value, NULL);		
	printf("OpenCL C Version: %s\n",value);
	free(value);

	cl_uint maxComputeUnits;
	ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);		
	checkErr(ret, "Compute Units");	
	printf("Compute units: %d\n",maxComputeUnits);
}

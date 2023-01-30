#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"

#define GROUP_SIZE 8

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program) {
  int filterSize = filterWidth * filterWidth * sizeof(float);
  int imageSize = imageHeight * imageWidth * sizeof(float);

  cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, NULL);

  cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterSize, filter, NULL);
  cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageSize, inputImage, NULL);
  cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);

  cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);
  clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_inputImage);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_outputImage);

  size_t global_work[] = {imageWidth, imageHeight};
  size_t work_group[] = {GROUP_SIZE, GROUP_SIZE};
  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work, work_group, 0, NULL, NULL);

  clEnqueueReadBuffer(command_queue, d_outputImage, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);

  clReleaseKernel(kernel);
  clReleaseMemObject(d_filter);
  clReleaseMemObject(d_inputImage);
  clReleaseMemObject(d_outputImage);
  clReleaseCommandQueue(command_queue);
}
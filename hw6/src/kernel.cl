__kernel void convolution(const int filterWidth,
                          __constant const float *restrict filter,
                          __global const float *restrict inputImage,
                          __global float *restrict outputImage) {
  int imageWidth = get_global_size(0);
  int imageHeight = get_global_size(1);
  int halfFilterSize = filterWidth >> 1;

  int x = get_global_id(0);
  int y = get_global_id(1);

  int row_offset_min = -min(y, halfFilterSize);
  int row_offset_max = min(imageHeight - 1 - y, halfFilterSize);
  int col_offset_min = -min(x, halfFilterSize);
  int col_offset_max = min(imageWidth - 1 - x, halfFilterSize);

  float sum = 0;
  for (int row_offset = row_offset_min; row_offset <= row_offset_max; row_offset++) {
    int imageBase = (y + row_offset) * imageWidth + x;
    int filterBase = (halfFilterSize + row_offset) * filterWidth + halfFilterSize;
    for (int col_offset = col_offset_min; col_offset <= col_offset_max; col_offset++) {
      if (filter[filterBase + col_offset]) {
        sum = mad(filter[filterBase + col_offset], inputImage[imageBase + col_offset], sum);
      }
    }
  }

  outputImage[y * imageWidth + x] = sum;
}
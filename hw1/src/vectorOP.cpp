#include "PPintrin.h"

__pp_mask ONE_MASK = _pp_init_ones();
__pp_mask ZERO_MASK = _pp_init_ones(0);
__pp_vec_int V_ZERO_I = _pp_vset_int(0);
__pp_vec_int V_ONE_I = _pp_vset_int(1);

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH) {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float v_values, v_result;
  __pp_vec_int v_exponents;
  __pp_mask values_mask; // values that are greater than 9.999999
  __pp_mask exponents_mask; // exponents that are greater than 0
  __pp_vec_float v_max = _pp_vset_float(9.999999f);

  int i = 0;
  while (i < N - VECTOR_WIDTH) {

    v_result = _pp_vset_float(1.0f);

    _pp_vload_float(v_values, values + i, ONE_MASK);
    _pp_vload_int(v_exponents, exponents + i, ONE_MASK);

    _pp_vgt_int(exponents_mask, v_exponents, V_ZERO_I, ONE_MASK);
    while (_pp_cntbits(exponents_mask) > 0) {
      _pp_vmult_float(v_result, v_result, v_values, exponents_mask);

      _pp_vsub_int(v_exponents, v_exponents, V_ONE_I, exponents_mask);
      _pp_vgt_int(exponents_mask, v_exponents, V_ZERO_I, exponents_mask);
    }

    _pp_vgt_float(values_mask, v_result, v_max, ONE_MASK);
    _pp_vmove_float(v_result, v_max, values_mask);

    _pp_vstore_float(output + i, v_result, ONE_MASK);

    i += VECTOR_WIDTH;
  }

  __pp_mask load_mask = _pp_init_ones(N - i);
  v_result = _pp_vset_float(1.0f);

  _pp_vload_float(v_values, values + i, load_mask);
  _pp_vload_int(v_exponents, exponents + i, load_mask);

  _pp_vgt_int(exponents_mask, v_exponents, V_ZERO_I, load_mask);
  while (_pp_cntbits(exponents_mask) > 0) {
    _pp_vmult_float(v_result, v_result, v_values, exponents_mask);

    _pp_vsub_int(v_exponents, v_exponents, V_ONE_I, exponents_mask);
    _pp_vgt_int(exponents_mask, v_exponents, V_ZERO_I, exponents_mask);
  }

  _pp_vgt_float(values_mask, v_result, v_max, load_mask);
  _pp_vmove_float(v_result, v_max, values_mask);

  _pp_vstore_float(output + i, v_result, load_mask);
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float result;
  __pp_mask result_mask = _pp_init_ones(1);
  __pp_vec_float v_values;
  __pp_vec_float v_result = _pp_vset_float(0.f);

  for (int i = 0; i < N; i += VECTOR_WIDTH) {
    _pp_vload_float(v_values, values + i, ONE_MASK);
    _pp_vadd_float(v_result, v_result, v_values, ONE_MASK);
  }

  int n = VECTOR_WIDTH;
  while (n > 1) {
    _pp_hadd_float(v_result, v_result);
    _pp_interleave_float(v_result, v_result);
    n /= 2;
  }

  _pp_vstore_float(&result, v_result, result_mask);
  return result;
}
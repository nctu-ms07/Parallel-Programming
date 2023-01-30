#include <bits/stdc++.h>

#ifndef __AVX2_AVAILABLE__
#define __AVX2_AVAILABLE__

#include <Xoshiro256Plus.h>
typedef SEFUtility::RNG::Xoshiro256Plus<SIMDInstructionSet::AVX2> Xoshiro256PlusAVX2;

#endif

using namespace std;

const __m256 rand_max = _mm256_set1_ps(RAND_MAX);
const __m256 one = _mm256_set1_ps(1.0f);

void *toss(void *number) {
  long long toss_num = *((long long *) number);
  long long in_circle_num = 0;

  Xoshiro256PlusAVX2 rng(rand());
  alignas(32) float result[8];

  for (long long i = 0; i < toss_num; i++) { // perform 8 toss at a time
    __m256 rand_float_x = _mm256_cvtepi32_ps(rng.next4().operator __m256i()); // convert random int to float
    __m256 float_x = _mm256_div_ps(rand_float_x, rand_max); // scale to -1 to 1

    __m256 rand_float_y = _mm256_cvtepi32_ps(rng.next4().operator __m256i()); // convert random int to float
    __m256 float_y = _mm256_div_ps(rand_float_y, rand_max); // scale to -1 to 1

    __m256 distance = _mm256_add_ps(_mm256_mul_ps(float_x, float_x), _mm256_mul_ps(float_y, float_y)); // x * x + y * y
    __m256 in_circle_mask = _mm256_cmp_ps(distance, one, _CMP_LE_OS); // distance <= 1

    __m256 in_circle = _mm256_and_ps(one, in_circle_mask); // a1, a2, a3, a4, a5, a6, a7, a8
    __m256 in_circle_permute = _mm256_permute2f128_ps(in_circle, in_circle, 1); // a5, a6, a7, a8, a1, a2, a3, a4

    in_circle = _mm256_hadd_ps(in_circle, in_circle_permute); // a1+a2, a3+a4, a5+a6, a7+a8, ....
    in_circle = _mm256_hadd_ps(in_circle, in_circle); // a1+a2+a3+a4, a5+a6+a7+a8, ....
    in_circle = _mm256_hadd_ps(in_circle, in_circle); // a1+a2+a3+a4+a5+a6+a7+a8, ....

    _mm256_store_ps(result, in_circle);
    // explicit conversion is important
    // long long 64-bit will be implicitly convert to float 32-bit if not specify
    in_circle_num += (short) result[0];
  }
  *((long long *) number) = in_circle_num;
  return nullptr;
}

int main(int argc, char *argv[]) {
  int thread_num = strtol(argv[1], nullptr, 10);
  long long total_toss = strtoll(argv[2], nullptr, 10);

  srand(time(nullptr));

  pthread_t threads[thread_num];
  long long *num[thread_num];

  long long quotient = ((total_toss / 8) / thread_num); // perform 8 toss at a time
  int remainder = ((total_toss / 8) % thread_num);
  for (int i = 0; i < thread_num; i++) {
    num[i] = new long long;
    *num[i] = quotient;
    if (remainder) {
      remainder--;
      (*num[i])++;
    }
    pthread_create(&threads[i], nullptr, toss, (void *) num[i]);
  }

  long long in_circle_num = 0;
  for (int i = 0; i < total_toss % 8; i++) {
    float x = rand() / ((float) RAND_MAX) * 2 - 1;
    float y = rand() / ((float) RAND_MAX) * 2 - 1;
    if (x * x + y * y <= 1) {
      in_circle_num++;
    }
  }

  for (int i = 0; i < thread_num; i++) {
    pthread_join(threads[i], nullptr);
    in_circle_num += *num[i];
    delete num[i];
  }

  cout << setprecision(10) << 4 * (in_circle_num / ((double) total_toss)) << '\n';
  return 0;
}


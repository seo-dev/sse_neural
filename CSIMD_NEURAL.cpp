void maxpool2x2_SSE(float* input, float* output, int width, int height) {
    // 입력 이미지의 너비와 높이를 고려하여 반복
    for (int i = 0; i < height; i += 2) {
        for (int j = 0; j < width; j += 2) {
            // 2x2 블록의 시작 인덱스 계산
            int idx = i * width + j;

            // SSE 레지스터에 데이터 로드
            __m128 block = _mm_set_ps(input[idx], input[idx + 1], input[idx + width], input[idx + width + 1]);

            // 최대값 연산
            block = _mm_max_ps(block, _mm_shuffle_ps(block, block, _MM_SHUFFLE(0, 0, 3, 2)));
            block = _mm_max_ps(block, _mm_shuffle_ps(block, block, _MM_SHUFFLE(0, 0, 0, 1)));

            // 출력 버퍼에 최대값 저장
            _mm_store_ss(&output[(i / 2) * (width / 2) + (j / 2)], block);
        }
    }
}

void neon_max_pooling(float32x4_t *input, int input_height, int input_width, int channels,
                      float32x4_t *output, int output_height, int output_width) {
  int kernel_size = 3;
  int stride = 2;
  int padding = 1;

  // 출력 크기 계산
  int output_size = output_height * output_width * channels;

  // 출력 요소 각각에 대해 루프
  for (int c = 0; c < channels; c++) {
    for (int y = 0; y < output_height; y++) {
      for (int x = 0; x < output_width; x++) {
        // 입력 좌표 계산
        int input_y = y * stride - padding;
        int input_x = x * stride - padding;

        // 입력 데이터에서 3x3 커널 로드
        float32x4_t kernel[3][3];
        for (int ky = 0; ky < kernel_size; ky++) {
          for (int kx = 0; kx < kernel_size; kx++) {
            int input_idx = (input_y + ky) * input_width * channels + (input_x + kx) * channels + c;
            kernel[ky][kx] = vld1q_f32(&input[input_idx]);
          }
        }

        // NEON 인스트럭션을 사용하여 최대 풀링 수행
        float32x4_t max_val = vdupq_n_f32(-FLT_MAX);
        for (int ky = 0; ky < kernel_size; ky++) {
          for (int kx = 0; kx < kernel_size; kx++) {
            max_val = vmaxq_f32(max_val, kernel[ky][kx]);
          }
        }

        // 최대 풀링된 값을 출력 배열에 저장
        int output_idx = y * output_width * channels + x * channels + c;
        vst1q_f32(&output[output_idx], max_val);
      }
    }
  }
}

void neon_conv2d(float32x4_t *input, int input_height, int input_width, int channels,
                 float32x4_t *weights, int kernel_size, int stride,
                 float32x4_t *output, int output_height, int output_width) {
  int padding = (kernel_size - 1) / 2;

  // 출력 크기 계산
  int output_size = output_height * output_width * channels;

  // 출력 요소 각각에 대해 루프
  for (int c = 0; c < channels; c++) {
    for (int y = 0; y < output_height; y++) {
      for (int x = 0; x < output_width; x++) {
        // 입력 좌표 계산
        int input_y = y * stride - padding;
        int input_x = x * stride - padding;

        // 가중치에서 3x3 커널 로드
        float32x4_t kernel[3][3];
        for (int ky = 0; ky < kernel_size; ky++) {
          for (int kx = 0; kx < kernel_size; kx++) {
            int weight_idx = (ky * kernel_size + kx) * channels + c;
            kernel[ky][kx] = vld1q_f32(&weights[weight_idx]);
          }
        }

        // 입력 패치 로드
        float32x4_t input_patch[3][3];
        for (int ky = 0; ky < kernel_size; ky++) {
          for (int kx = 0; kx < kernel_size; kx++) {
            int input_idx = (input_y + ky) * input_width * channels + (input_x + kx) * channels + c;
            input_patch[ky][kx] = vld1q_f32(&input[input_idx]);
          }
        }

        // NEON 명령어를 사용하여畳み込み 수행
        float32x4_t sum = vdupq_n_f32(0.0f);
        for (int ky = 0; ky < kernel_size; ky++) {
          for (int kx = 0; kx < kernel_size; kx++) {
            sum = vfmaq_f32(sum, kernel[ky][kx], input_patch[ky][kx]);
          }
        }

        // 출력 값을 저장
        int output_idx = y * output_width * channels + x * channels + c;
        vst1q_f32(&output[output_idx], sum);
      }
    }
  }
}

void maxpool_std(const float* input, float* output, int input_width, int input_height) {
    // Output dimensions
    int output_width = (input_width - 3) / 2 + 1;
    int output_height = (input_height - 3) / 2 + 1;

    // Iterate over each output element
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            // Calculate input indices
            int input_x = x * 2;
            int input_y = y * 2;

            // Find maximum value in 3x3 region
            float max_val = input[input_y * input_width + input_x];
            for (int dy = 0; dy < 3; ++dy) {
                for (int dx = 0; dx < 3; ++dx) {
                    max_val = std::max(max_val, input[(input_y + dy) * input_width + input_x + dx]);
                }
            }

            // Store the maximum value in output
            output[y * output_width + x] = max_val;
        }
    }
}
void maxpool_sse(const float* input, float* output, int input_width, int input_height) {
    // Output dimensions
    int output_width = (input_width - 3) / 2 + 1;
    int output_height = (input_height - 3) / 2 + 1;

    // Iterate over each output element
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            // Calculate input indices
            int input_x = x * 2;
            int input_y = y * 2;

            // Load 3x3 region into SIMD registers
            __m128 v0 = _mm_loadu_ps(&input[(input_y + 0) * input_width + input_x]);
            __m128 v1 = _mm_loadu_ps(&input[(input_y + 1) * input_width + input_x]);
            __m128 v2 = _mm_loadu_ps(&input[(input_y + 2) * input_width + input_x]);

            // Combine adjacent pairs to find maximums
            __m128 max1 = _mm_max_ps(v0, v1);
            __m128 max2 = _mm_max_ps(max1, v2);

            // Shuffle to get the maximums of each pair
            __m128 shuf = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(3, 1, 2, 0));
            __m128 max3 = _mm_max_ps(max2, shuf);

            // Store the result in output
            _mm_storeu_ps(&output[y * output_width + x], max3);
        }
    }
}

void maxpool3x3_SSE(float* input, float* output, int width, int height) {
    // 입력 이미지의 너비와 높이를 고려하여 반복
    for (int i = 0; i <= height - 3; i += 3) {
        for (int j = 0; j <= width - 3; j += 3) {
            // 3x3 블록의 시작 인덱스 계산
            int idx = i * width + j;
            
            // 첫 번째 3개의 원소 로드
            __m128 row1 = _mm_loadu_ps(&input[idx]);
            __m128 row2 = _mm_loadu_ps(&input[idx + width]);
            __m128 row3 = _mm_loadu_ps(&input[idx + 2 * width]);

            // 각 행의 첫 3개 원소에서 최대값을 구함
            __m128 max1 = _mm_max_ps(row1, row2);
            max1 = _mm_max_ps(max1, row3);

            // 각 행의 나머지 원소를 고려하여 최대값 업데이트
            __m128 tail1 = _mm_loadu_ps(&input[idx + 3]);
            __m128 tail2 = _mm_loadu_ps(&input[idx + width + 3]);
            __m128 tail3 = _mm_loadu_ps(&input[idx + 2 * width + 3]);

            __m128 max2 = _mm_max_ps(tail1, tail2);
            max2 = _mm_max_ps(max2, tail3);

            // 최종 최대값 계산 (9개 원소 중 최대값)
            __m128 final_max = _mm_max_ps(max1, max2);
            final_max = _mm_max_ps(_mm_shuffle_ps(final_max, final_max, _MM_SHUFFLE(0, 0, 3, 2)),
                                   _mm_shuffle_ps(final_max, final_max, _MM_SHUFFLE(0, 0, 0, 1)));

            // 출력 버퍼에 최대값 저장
            _mm_store_ss(&output[(i / 3) * (width / 3) + (j / 3)], final_max);
        }
    }
}

void maxpool3x3_stride2_SSE(float* input, float* output, int width, int height) {
    // 입력 이미지의 너비와 높이를 고려하여 반복 (스트라이드 2 적용)
    for (int i = 0; i <= height - 3; i += 2) {
        for (int j = 0; j <= width - 3; j += 2) {
            // 3x3 블록의 시작 인덱스 계산
            int idx = i * width + j;
            
            // 첫 번째 3개의 원소 로드
            __m128 row1 = _mm_loadu_ps(&input[idx]);
            __m128 row2 = _mm_loadu_ps(&input[idx + width]);
            __m128 row3 = _mm_loadu_ps(&input[idx + 2 * width]);

            // 각 행의 최대값을 구함
            __m128 max1 = _mm_max_ps(row1, row2);
            max1 = _mm_max_ps(max1, row3);

            // 최종 최대값 계산 (3x3에서 최대값)
            max1 = _mm_max_ps(_mm_shuffle_ps(max1, max1, _MM_SHUFFLE(0, 0, 3, 2)),
                              _mm_shuffle_ps(max1, max1, _MM_SHUFFLE(0, 0, 0, 1)));

            // 출력 버퍼에 최대값 저장
            _mm_store_ss(&output[(i / 2) * (width / 2) + (j / 2)], max1);
        }
    }
}

void maxPooling3x3SSE(float* input, float* output, int width, int height) {
    int outWidth = width / 2;
    int outHeight = height / 2;

    for (int y = 0; y < height - 1; y += 2) {
        for (int x = 0; x < width - 1; x += 2) {
            // Load 9 pixels (3x3 region)
            __m128 row1 = _mm_loadu_ps(&input[y * width + x]);
            __m128 row2 = _mm_loadu_ps(&input[(y + 1) * width + x]);
            __m128 row3 = _mm_loadu_ps(&input[(y + 2) * width + x]);

            // Get max of first 3 elements in each row
            __m128 max1 = _mm_max_ps(_mm_shuffle_ps(row1, row1, _MM_SHUFFLE(1, 0, 3, 2)), row1);
            max1 = _mm_max_ps(_mm_shuffle_ps(max1, max1, _MM_SHUFFLE(2, 1, 0, 3)), max1);

            __m128 max2 = _mm_max_ps(_mm_shuffle_ps(row2, row2, _MM_SHUFFLE(1, 0, 3, 2)), row2);
            max2 = _mm_max_ps(_mm_shuffle_ps(max2, max2, _MM_SHUFFLE(2, 1, 0, 3)), max2);

            __m128 max3 = _mm_max_ps(_mm_shuffle_ps(row3, row3, _MM_SHUFFLE(1, 0, 3, 2)), row3);
            max3 = _mm_max_ps(_mm_shuffle_ps(max3, max3, _MM_SHUFFLE(2, 1, 0, 3)), max3);

            // Get the final max from all rows
            __m128 finalMax = _mm_max_ps(_mm_max_ps(max1, max2), max3);

            // Store the result in output
            _mm_store_ss(&output[(y / 2) * outWidth + (x / 2)], finalMax);
        }
    }
}

// const int width = 10;
// const int height = 10;
// float input[100];
// for (int i = 0; i < 100; i++) input[i] = float(i);

// int kernel_size = 3;
// int stride = 2;
// int outputWidth = (width - kernel_size) / stride + 1;
// int outputHeight = (height - kernel_size) / stride + 1;
// float* output = new float[outputWidth * outputHeight];

// // Maxpooling 실행
// maxpool_SSE(input, output, width, height, kernel_size, stride);

// // 결과 출력
// for (int i = 0; i < outputHeight; i++) {
//     for (int j = 0; j < outputWidth; j++) {
//         std::cout << output[i * outputWidth + j] << " ";
//     }
//     std::cout << std::endl;
// }

// delete[] output;  // 할당된 메모리 해제


void dynamic_maxpool_SSE(float* input, float* output, int width, int height, int kernel_size, int stride) {
    int outputWidth = (width - kernel_size) / stride + 1;
    int outputHeight = (height - kernel_size) / stride + 1;

    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            __m128 maxVal = _mm_set1_ps(-FLT_MAX);  // 최소값으로 maxVal 초기화
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; kx += 4) {  // 4개 단위로 처리
                    int idx = ((y * stride + ky) * width + (x * stride + kx));
                    __m128 elem = _mm_loadu_ps(&input[idx]);  // unaligned load
                    maxVal = _mm_max_ps(maxVal, elem);
                }
            }
            // 4개 중 최대값을 스칼라로 추출
            float mvals[4];
            _mm_storeu_ps(mvals, maxVal);
            float finalMax = mvals[0];
            for (int i = 1; i < 4; i++) {
                if (mvals[i] > finalMax) finalMax = mvals[i];
            }
            output[y * outputWidth + x] = finalMax;

            // // 4개 중 최대값을 스칼라로 추출
            // float mvals[4];
            // _mm_storeu_ps(mvals, maxVal);
            // float finalMax = std::max(std::max(mvals[0], mvals[1]), std::max(mvals[2], mvals[3]));
            // output[y * outputWidth + x] = finalMax;
        }
    }
}
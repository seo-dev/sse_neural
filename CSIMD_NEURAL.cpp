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
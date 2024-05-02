void maxpool_SSE(float* input, float* output, int width, int height) {
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
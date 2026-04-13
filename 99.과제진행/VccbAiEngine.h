cat > ~/vccb_ai/VccbAiEngine.h << 'EOF'
// ============================================================
// VccbAiEngine.h
// GRU 순전파 추론 엔진 (C++ 직접 포팅)
// ============================================================
#pragma once
#include <cmath>

// 입출력 구조체
struct VccbAiInput {
    float seq[10][2];   // 시퀀스 Feature [SEQ_LEN][SEQ_INPUT]
    float stat[15];     // 정적 Feature   [STATIC_INPUT]
};

struct VccbAiOutput {
    float resume_prob;  // Resume 확률 (0.0 ~ 1.0)
    float resume_time;  // 예상 Resume 시간 (초)
};

// 추론 엔진 함수
void  VccbAiEngine_Init();
bool  VccbAiEngine_Infer(const VccbAiInput& input, VccbAiOutput& output);
EOF

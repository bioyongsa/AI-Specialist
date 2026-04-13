cat > ~/vccb_ai/test_engine.cpp << 'EOF'
#include <cstdio>
#include "VccbAiEngine.h"

int main()
{
    VccbAiEngine_Init();

    VccbAiInput input = {};

    // 테스트용 더미 입력
    for (int t = 0; t < 10; t++) {
        input.seq[t][0] = 30.0f;   // duration (초)
        input.seq[t][1] = 1.0f;    // resume_cause
    }
    for (int i = 0; i < 15; i++) {
        input.stat[i] = 0.5f;
    }

    VccbAiOutput output = {};
    bool ok = VccbAiEngine_Infer(input, output);

    if (ok) {
        printf("추론 성공!\n");
        printf("  resume_prob: %.4f\n", output.resume_prob);
        printf("  resume_time: %.4f 초\n", output.resume_time);
    } else {
        printf("추론 실패\n");
    }

    return 0;
}
EOF

cat > ~/vccb_ai/VccbAiEngine.cpp << 'EOF'
// ============================================================
// VccbAiEngine.cpp
// GRU 순전파 추론 엔진 (C++ 직접 포팅)
// ============================================================
#include "VccbAiEngine.h"
#include "VccbAiWeights.h"
#include <cstring>
#include <cmath>

using namespace VccbAiWeights;

// ──────────────────────────────────────────
// 내부 유틸 함수
// ──────────────────────────────────────────

// sigmoid
static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// tanh
static inline float tanhf_wrap(float x)
{
    return tanhf(x);
}

// relu
static inline float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

// 벡터 내적 + bias
// result = weight[row] · vec + bias[row]
static float dot(const float* weight, const float* vec,
                 int vec_size, int row)
{
    float sum = 0.0f;
    const float* w = weight + row * vec_size;
    for (int i = 0; i < vec_size; i++) {
        sum += w[i] * vec[i];
    }
    return sum;
}

// ──────────────────────────────────────────
// GRU 단일 스텝
// ──────────────────────────────────────────
// PyTorch GRU 게이트 순서: [reset | update | new]
// r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
// z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
// n = tanh   (W_in * x + b_in + r * (W_hn * h + b_hn))
// h = (1 - z) * n + z * h
static void gru_step(const float* x,    // 입력 [SEQ_INPUT]
                     float*       h,    // 은닉 상태 [HIDDEN] (in/out)
                     int          input_size,
                     int          hidden_size)
{
    // 게이트별 오프셋
    const int R = 0;
    const int Z = hidden_size;
    const int N = hidden_size * 2;

    // reset gate
    float r[16] = {};
    for (int i = 0; i < hidden_size; i++) {
        float val = dot(GRU_WEIGHT_IH, x, input_size, R + i) + GRU_BIAS_IH[R + i]
                  + dot(GRU_WEIGHT_HH, h, hidden_size, R + i) + GRU_BIAS_HH[R + i];
        r[i] = sigmoid(val);
    }

    // update gate
    float z[16] = {};
    for (int i = 0; i < hidden_size; i++) {
        float val = dot(GRU_WEIGHT_IH, x, input_size, Z + i) + GRU_BIAS_IH[Z + i]
                  + dot(GRU_WEIGHT_HH, h, hidden_size, Z + i) + GRU_BIAS_HH[Z + i];
        z[i] = sigmoid(val);
    }

    // new gate
    float n[16] = {};
    for (int i = 0; i < hidden_size; i++) {
        float hh = dot(GRU_WEIGHT_HH, h, hidden_size, N + i) + GRU_BIAS_HH[N + i];
        float val = dot(GRU_WEIGHT_IH, x, input_size, N + i) + GRU_BIAS_IH[N + i]
                  + r[i] * hh;
        n[i] = tanhf_wrap(val);
    }

    // 은닉 상태 업데이트
    for (int i = 0; i < hidden_size; i++) {
        h[i] = (1.0f - z[i]) * n[i] + z[i] * h[i];
    }
}

// ──────────────────────────────────────────
// 초기화
// ──────────────────────────────────────────
void VccbAiEngine_Init()
{
    // 현재는 별도 초기화 불필요
    // 가중치는 VccbAiWeights.h에 상수로 내장됨
}

// ──────────────────────────────────────────
// 추론
// ──────────────────────────────────────────
bool VccbAiEngine_Infer(const VccbAiInput& input, VccbAiOutput& output)
{
    const int hidden  = HIDDEN_SIZE;
    const int seq_in  = SEQ_INPUT_SIZE;
    const int stat_in = STATIC_INPUT_SIZE;
    const int seq_len = SEQ_LEN;

    // ── GRU 순전파 ──
    float h[16] = {};   // 은닉 상태 초기화 (0)

    for (int t = 0; t < seq_len; t++) {
        gru_step(input.seq[t], h, seq_in, hidden);
    }
    // h = GRU 최종 출력 [hidden]

    // ── MLP 순전파 ──
    float mlp_out[16] = {};
    for (int i = 0; i < hidden; i++) {
        float val = dot(MLP_WEIGHT, input.stat, stat_in, i) + MLP_BIAS[i];
        mlp_out[i] = relu(val);
    }

    // ── 결합 ──
    // combined = [h(0..15), mlp_out(0..15)]  크기: hidden*2
    float combined[32] = {};
    for (int i = 0; i < hidden; i++) {
        combined[i]          = h[i];
        combined[i + hidden] = mlp_out[i];
    }

    // ── Output Head ──
    int combined_size = hidden * 2;

    // resume_prob
    float prob_val = dot(HEAD_PROB_WEIGHT, combined, combined_size, 0)
                   + HEAD_PROB_BIAS[0];
    output.resume_prob = sigmoid(prob_val);

    // resume_time
    float time_val = dot(HEAD_TIME_WEIGHT, combined, combined_size, 0)
                   + HEAD_TIME_BIAS[0];
    output.resume_time = time_val;

    return true;
}
EOF

cat > ~/vccb_ai/export_weights.py << 'EOF'
import torch
import torch.nn as nn
import numpy as np

# ══════════════════════════════════════════════
# 1. 모델 구조 (train_gru.py와 동일해야 함)
# ══════════════════════════════════════════════
class RRCStatePredictor(nn.Module):
    def __init__(self, seq_input_size, static_input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(static_input_size, hidden_size),
            nn.ReLU()
        )
        combined = hidden_size * 2
        self.head_prob = nn.Linear(combined, 1)
        self.head_time = nn.Linear(combined, 1)

    def forward(self, seq, static):
        _, h_n = self.gru(seq)
        gru_out = h_n.squeeze(0)
        mlp_out = self.mlp(static)
        combined = torch.cat([gru_out, mlp_out], dim=1)
        prob = torch.sigmoid(self.head_prob(combined))
        time = self.head_time(combined)
        return prob, time


# ══════════════════════════════════════════════
# 2. 가중치 추출 함수
# ══════════════════════════════════════════════
def tensor_to_c_array(tensor, name):
    """텐서를 C++ 배열 선언 문자열로 변환"""
    arr = tensor.detach().numpy().flatten()
    size = len(arr)
    values = ", ".join(f"{v:.8f}f" for v in arr)
    return f"static const float {name}[{size}] = {{\n    {values}\n}};\n"


def export_weights(model, hidden_size, seq_input, static_input):
    sd = model.state_dict()

    lines = []
    lines.append("// ============================================================")
    lines.append("// VccbAiWeights.h")
    lines.append("// 자동 생성 파일 — export_weights.py로만 수정할 것")
    lines.append("// ============================================================")
    lines.append("#pragma once")
    lines.append("")
    lines.append("namespace VccbAiWeights {")
    lines.append("")

    # 상수 정의
    lines.append(f"static const int HIDDEN_SIZE    = {hidden_size};")
    lines.append(f"static const int SEQ_INPUT_SIZE = {seq_input};")
    lines.append(f"static const int STATIC_INPUT_SIZE = {static_input};")
    lines.append(f"static const int SEQ_LEN        = 10;")
    lines.append("")

    # GRU 가중치
    # PyTorch GRU 가중치 순서: [reset | update | new]
    # weight_ih_l0: [3*hidden, input]
    # weight_hh_l0: [3*hidden, hidden]
    # bias_ih_l0:   [3*hidden]
    # bias_hh_l0:   [3*hidden]
    lines.append("// GRU 가중치")
    lines.append(tensor_to_c_array(sd["gru.weight_ih_l0"], "GRU_WEIGHT_IH"))
    lines.append(tensor_to_c_array(sd["gru.weight_hh_l0"], "GRU_WEIGHT_HH"))
    lines.append(tensor_to_c_array(sd["gru.bias_ih_l0"],   "GRU_BIAS_IH"))
    lines.append(tensor_to_c_array(sd["gru.bias_hh_l0"],   "GRU_BIAS_HH"))

    # MLP 가중치 (mlp.0 = Linear)
    lines.append("// MLP 가중치")
    lines.append(tensor_to_c_array(sd["mlp.0.weight"], "MLP_WEIGHT"))
    lines.append(tensor_to_c_array(sd["mlp.0.bias"],   "MLP_BIAS"))

    # Output Head 가중치
    lines.append("// Output Head 가중치")
    lines.append(tensor_to_c_array(sd["head_prob.weight"], "HEAD_PROB_WEIGHT"))
    lines.append(tensor_to_c_array(sd["head_prob.bias"],   "HEAD_PROB_BIAS"))
    lines.append(tensor_to_c_array(sd["head_time.weight"], "HEAD_TIME_WEIGHT"))
    lines.append(tensor_to_c_array(sd["head_time.bias"],   "HEAD_TIME_BIAS"))

    lines.append("} // namespace VccbAiWeights")

    return "\n".join(lines)


# ══════════════════════════════════════════════
# 3. 실행
# ══════════════════════════════════════════════
if __name__ == "__main__":
    HIDDEN_SIZE  = 16
    SEQ_INPUT    = 2
    STATIC_INPUT = 15

    # 모델 로드
    model = RRCStatePredictor(SEQ_INPUT, STATIC_INPUT, HIDDEN_SIZE)
    model.load_state_dict(torch.load("gru_model.pth"))
    model.eval()
    print("모델 로드 완료")

    # 헤더 파일 생성
    header_code = export_weights(model, HIDDEN_SIZE, SEQ_INPUT, STATIC_INPUT)

    with open("VccbAiWeights.h", "w") as f:
        f.write(header_code)

    print("VccbAiWeights.h 생성 완료!")

    # 생성된 파일 크기 확인
    import os
    size = os.path.getsize("VccbAiWeights.h")
    print(f"파일 크기: {size} bytes")

    # 가중치 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params}")
EOF

cat > ~/vccb_ai/train_gru.py << 'EOF'
import torch
import torch.nn as nn
import numpy as np

# ══════════════════════════════════════════════
# 1. 모델 구조 정의
# ══════════════════════════════════════════════
class RRCStatePredictor(nn.Module):
    def __init__(self, seq_input_size, static_input_size, hidden_size):
        super().__init__()

        # GRU: 시퀀스 Feature 처리 (과거 이력)
        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # MLP: 정적 Feature 처리
        self.mlp = nn.Sequential(
            nn.Linear(static_input_size, hidden_size),
            nn.ReLU()
        )

        # Output Head
        combined = hidden_size * 2
        self.head_prob = nn.Linear(combined, 1)   # resume 확률
        self.head_time = nn.Linear(combined, 1)   # resume 예상 시간

    def forward(self, seq, static):
        _, h_n = self.gru(seq)
        gru_out = h_n.squeeze(0)

        mlp_out = self.mlp(static)

        combined = torch.cat([gru_out, mlp_out], dim=1)

        prob = torch.sigmoid(self.head_prob(combined))
        time = self.head_time(combined)

        return prob, time


# ══════════════════════════════════════════════
# 2. 더미 데이터 생성 (실제 로그 데이터 대신)
# ══════════════════════════════════════════════
def make_dummy_data(n_samples=1000):
    SEQ_LEN       = 10
    SEQ_INPUT     = 2    # [duration, resume_cause]
    STATIC_INPUT  = 15   # 정적 Feature 수

    seq    = torch.randn(n_samples, SEQ_LEN, SEQ_INPUT)
    static = torch.randn(n_samples, STATIC_INPUT)

    # 레이블: resume 확률 (0/1), resume 시간 (초)
    label_prob = torch.randint(0, 2, (n_samples, 1)).float()
    label_time = torch.rand(n_samples, 1) * 60.0

    return seq, static, label_prob, label_time


# ══════════════════════════════════════════════
# 3. 학습
# ══════════════════════════════════════════════
def train():
    HIDDEN_SIZE   = 16
    SEQ_INPUT     = 2
    STATIC_INPUT  = 15
    EPOCHS        = 50
    LR            = 0.001

    model = RRCStatePredictor(SEQ_INPUT, STATIC_INPUT, HIDDEN_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_bce  = nn.BCELoss()
    loss_mse  = nn.MSELoss()

    seq, static, label_prob, label_time = make_dummy_data(1000)

    print("학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        pred_prob, pred_time = model(seq, static)

        loss = loss_bce(pred_prob, label_prob) + \
               0.1 * loss_mse(pred_time, label_time)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} / {EPOCHS}  Loss: {loss.item():.4f}")

    print("학습 완료!")

    # 모델 저장
    torch.save(model.state_dict(), "gru_model.pth")
    print("모델 저장 완료: gru_model.pth")

    return model, HIDDEN_SIZE, SEQ_INPUT, STATIC_INPUT


if __name__ == "__main__":
    train()
EOF

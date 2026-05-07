class PreNorm(nn.Module):  # Transformer의 Pre-LN 구조: LayerNorm 후 블록 실행
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 토큰 임베딩 차원(dim) 기준 LayerNorm -> 패치별 정규화
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # 정규화된 토큰을 Attention/FFN에 전달

class FeedForward(nn.Module):  # Transformer의 FFN(MLP) 블록 -> 패치 자체의 정보를 더 깊게 분석
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(  # FFN: Linear→GELU→Dropout→Linear→Dropout 구성
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 전체 헤드 차원 = head 수 × head 차원 [SG] !!!!
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  # 멀티헤드 개수 저장
        self.scale = dim_head ** -0.5  # Scaled dot-product를 위한 스케일(1/√d)

        self.attend = nn.Softmax(dim = -1)  # attention score를 확률로 변환(softmax), 마지막 차원 : sequence length N
        self.last_attn = None  # (학습/추론 시) 마지막 forward에서의 attention map 저장용(시각화/분석)  # attention score를 확률로 변환(softmax), 마지막 차원 : sequence length N
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 입력 토큰→Q,K,V를 한 번에 선형 변환 [SG] RGB 3개의 head를 한번에!!!

        self.to_out = nn.Sequential(  # 헤드들을 합친 뒤 출력 투영 + 드롭아웃
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads   # x = [batch Size, tokens, embedding dimension]
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 선형변환 결과를 Q,K,V로 분할 -> (Q, K ,V)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)  # multi-head 연산을 위해 각 head 기준 n x d 로 분리

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # 각 토큰 간 유사도(Q·Kᵀ) 계산 후 스케일 적용

        attn = self.attend(dots)  # 토큰 간 가중치(attention map) 생성
        self.last_attn = attn.detach()  # 그래프에서 분리(detach)해서 저장 (시각화 목적)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # attention 가중합으로 새로운 토큰 표현(out) 계산
        out = rearrange(out, 'b h n d -> b n (h d)')  # 헤드 차원을 다시 합쳐 (batch, tokens, dim)로 복원
        return self.to_out(out)  # 최종 투영을 거쳐 Attention 블록 출력 반환

class Transformer(nn.Module):  # Encoder block을 여러 층(depth) 쌓는 Transformer
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.): # [SG] __init__: 요리의 재료
        super().__init__()
        self.layers = nn.ModuleList([])  # 각 층(Attention+FFN)을 담을 컨테이너
        for _ in range(depth):  # depth 만큼 Encoder block 반복 생성
            self.layers.append(nn.ModuleList([  # 한 층 = (PreNorm+Attention) + (PreNorm+FFN)
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), # [SG] pre-norm 하고, attention해라
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)) # [SG] pre-norm 하고, feedforward해라!
            ]))
    def forward(self, x): # [SG] __forward__: 레시피
        for attn, ff in self.layers:
            x = attn(x) + x  # Residual 연결: Attention 결과를 입력에 더해 정보 보존 [SG] skip connection의 구조
            x = ff(x) + x  # Residual 연결: FFN 결과를 입력에 더해 정보 보존
        return x

class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        ...
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = cfg.channels * patch_height * patch_width
        ...
        self.to_patch_embedding = nn.Sequential(
            # 패치 그리드로 자른 뒤, 각 패치를 1D 벡터로 펼쳐 토큰 시퀀스를 만듦
            ...
            # 펼친 패치 벡터를 Transformer hidden size(dim)로 선형 투영
            nn.Linear(patch_dim, cfg.dim),
        )

        # 6) CLS 토큰 + 위치 임베딩(학습 파라미터)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, cfg.dim))
        self.dropout = nn.Dropout(cfg.emb_dropout)

        # 7) Transformer Encoder 스택(Attention + FFN + Residual/PreNorm)
        self.transformer = Transformer(
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
        )

        self.pool = cfg.pool

        # 8) 분류 헤드: (CLS/mean) 표현 -> LayerNorm -> Linear(logits)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.num_classes),
        )

    def forward(self, img): # [SG] !!!!
        # A) 패치 임베딩으로 토큰 시퀀스 생성: (B,C,H,W) -> (B,N,dim)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # B) CLS 토큰을 배치만큼 복제해 시퀀스 맨 앞에 붙임: (B,1,dim)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, dim)

        # C) 위치 임베딩을 더해 토큰 순서(공간 위치) 정보를 주입
        x = x + self.pos_embedding[:, : (n + 1)]   # (B, N+1, dim) + (1, N+1, dim) -> Broadcasting
        x = self.dropout(x)

        # D) Encoder를 통과하며 전역 의존성(Self-Attention) 학습
        x = self.transformer(x)  # (B, N+1, dim)

        # E) 이미지 표현 벡터 선택: CLS 토큰(0번) 또는 mean pooling
        x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)   # (B, dim)

        # F) 최종 logits 출력 (softmax는 CrossEntropyLoss 내부에서 처리)
        return self.mlp_head(x)

def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    n_steps = len(data_loader)
    log_steps = { ... }

    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)  # Softmax
        loss = F.nll_loss(output, target)           # Negative Log Likelihood
        loss.backward()
        optimizer.step()

        if i in log_steps:
            seen = min((i + 1) * len(data), total_samples)
            loss_history.append(loss.item())

def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += float(loss.item())
            correct_samples += int(pred.eq(target).sum().item())

    avg_loss = total_loss / total_samples
    acc = correct_samples / total_samples

    loss_history.append(avg_loss)
    return avg_loss, acc

# 3) U-Net 모델 정의
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        if mid_channels is None:
            mid_channels = out_channels

        self.net = nn.Sequential(
            # 1) (C_in → C_mid)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),

            # 2) (C_mid → C_out)
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)           # (B, C_in, H/2, W/2)
        x = self.conv(x)           # (B, C_out, H/2, W/2)
        return x

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)  # bilinear: 채널 유지 / deconv: 채널이 절반으로 감소
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,  # left, right
                        diff_y // 2, diff_y - diff_y // 2]) # top, bottom
        x = torch.cat([x2, x1], dim=1)  # (B, C_skip + C_up, H, W) == (B, in_channels, H, W)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)   # (B, n_channels, H, W) → (B, 64, H, W)
        self.down1 = Down(64, 128)              # → (B, 128, H/2, W/2)
        self.down2 = Down(128, 256)             # → (B, 256, H/4, W/4)
        self.down3 = Down(256, 512)             # → (B, 512, H/8, W/8)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # → (B, 1024/f, H/16, W/16)

        self.up1 = Up(1024, 512 // factor, bilinear)  # (skip=512)와 concat을 고려
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)       # (B, 64, H, W)
        x2 = self.down1(x1)    # (B, 128, H/2, W/2)
        x3 = self.down2(x2)    # (B, 256, H/4, W/4)
        x4 = self.down3(x3)    # (B, 512, H/8, W/8)
        x5 = self.down4(x4)    # (B, 1024/f, H/16, W/16)

        x = self.up1(x5, x4)   # (B, 512/f, H/8, W/8)
        x = self.up2(x, x3)    # (B, 256/f, H/4, W/4)
        x = self.up3(x, x2)    # (B, 128/f, H/2, W/2)
        x = self.up4(x, x1)    # (B, 64, H, W)

        logits = self.outc(x)  # (B, n_classes, H, W)
        return logits

model = UNet(n_channels=1, n_classes=1).to(device)







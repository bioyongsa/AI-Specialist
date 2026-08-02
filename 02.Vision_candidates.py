02.ViT_CIFRA10 # ------------------------------------------------------------------------------------
#5
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

class Attention(nn.Module):  # 멀티헤드 Self-Attention(MHSA) 구현
    def __init__(self, dim, heads = 4, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads  # 전체 헤드 차원 = head 수 × head 차원
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads  # 멀티헤드 개수 저장
        self.scale = dim_head ** -0.5  # Scaled dot-product를 위한 스케일(1/√d)
        self.attend = nn.Softmax(dim = -1)  # attention score를 확률로 변환(softmax), 마지막 차원 : sequence length N
        self.last_attn = None  # (학습/추론 시) 마지막 forward에서의 attention map 저장용(시각화/분석)  # attention score를 확률로 변환(softmax), 마지막 차원 : sequence length N
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 입력 토큰→Q,K,V를 한 번에 선형 변환

        self.to_out = nn.Sequential(  # 헤드들을 합친 뒤 출력 투영 + 드롭아웃
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        ...
        attn = self.attend(dots)  # 토큰 간 가중치(attention map) 생성
        self.last_attn = attn.detach()  # 그래프에서 분리(detach)해서 저장 (시각화 목적)
        ...        
        return self.to_out(out)  # 최종 투영을 거쳐 Attention 블록 출력 반환

class Transformer(nn.Module):  # Encoder block을 여러 층(depth) 쌓는 Transformer
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 각 층(Attention+FFN)을 담을 컨테이너
        for _ in range(depth):  # depth 만큼 Encoder block 반복 생성
            self.layers.append(nn.ModuleList([  # 한 층 = (PreNorm+Attention) + (PreNorm+FFN)
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # Residual 연결: Attention 결과를 입력에 더해 정보 보존
            x = ff(x) + x  # Residual 연결: FFN 결과를 입력에 더해 정보 보존
        return x

#6
class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        ...

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = cfg.channels * patch_height * patch_width

        ...

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, cfg.dim),
        )

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

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.num_classes),
        )

    def forward(self, img):
        ...
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

#10
def train_epoch(model, optimizer, data_loader, loss_history):
    model.train()
    ...

    for i, (data, target) in enumerate(data_loader):
        ...
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)  # Softmax
        loss = F.nll_loss(output, target)           # Negative Log Likelihood
        loss.backward()
        optimizer.step()

#11
def evaluate(model, data_loader, loss_history):
    model.eval()
    ...

    with torch.no_grad():
        for data, target in data_loader:
            ...
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += float(loss.item())
            correct_samples += int(pred.eq(target).sum().item())

    avg_loss = total_loss / total_samples
    acc = correct_samples / total_samples

    loss_history.append(avg_loss)
    return avg_loss, acc


03.DETR # ------------------------------------------------------------------------------------
#16
class AttentionVisualizer:  # DETR 내부 어텐션을 시각화하는 유틸 클래스 정의
    ...

    def compute_features(self, img):  # hook을 등록해 feature/attention을 수집한 뒤 forward 수행
        ....

        self.conv_features = conv_features[0]  # 리스트에서 실제 텐서/NestedTensor 꺼내기
        self.dec_attn_weights = dec_attn_weights[0]
        shape = self.conv_features['0'].tensors.shape[-2:]
        self.enc_attn_weights = enc_attn_weights[0].reshape(shape + shape)

    def compute_on_image(self, url):
        if url != self.url:
            ...
            self.tensor_img = self.transform(self.pil_img).unsqueeze(0)  # 입력 이미지 전처리 함수 보관
            self.compute_features(self.tensor_img)


04.Unet # ------------------------------------------------------------------------------------
#3
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        ...
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        ...

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
        super().__init__()

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

#5
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in dataloader:
        ...

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for imgs, masks in dataloader:
        ...

        logits = model(imgs)
        loss = criterion(logits, masks)
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss
    
05.DDPM # ------------------------------------------------------------------------------------
# 3
class SinusoidalPositionEmbeddings(nn.Module):    
    ...
    def forward(self, time):
        ...
        args = time[:, None].float() * freq[None, :]
        
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out) # 채널을 그룹으로 나눠 정규화 (배치 크기에 무관하게 안정적)
        self.act = nn.SiLU() # 최근 생성 모델에서 자주 쓰이는 매끄러운 활성화 함수

    def forward(self, x, scale_shift=None):
        x = self.proj(x)    # (B, dim, H, W) -> (B, dim_out, H, W)
        x = self.norm(x)    # shape 유지

        if exists(scale_shift):
            scale, shift = scale_shift  # scale:(B, C, 1, 1), shift:(B, C, 1, 1)
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

# 5
class Unet(nn.Module):
    def __init__(
            ...
    ):
        ...

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim), # 1. 숫자를 사인/코사인 파동 벡터로 변환
            nn.Linear(dim, time_dim),          # 2. 차원 확장
            nn.GELU(),                         # 3. 활성화 함수
            nn.Linear(time_dim, time_dim),     # 4. 최종 임베딩 벡터 완성
        )

        # 4. Downsampling Path (인코더)
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        ...

        # 5. Middle Block (최하단 보틀넥)
        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim))) # 여기서 정밀한 표준 어텐션 사용
        self.mid_block2 = block(mid_dim, mid_dim, time_emb_dim=time_dim)

        ...
        # 7. 최종 출력층
        self.out_dim = default(out_dim, channels)
        self.final_res_block = block(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        ...

        # [4] Middle (보틀넥: 가장 추상화된 특징 연산)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # [5] Up path (이미지를 다시 키우며 Skip Connection 결합)
        for block1, block2, attn, upsample in self.ups:
            # h.pop()으로 가장 나중에 저장된(가장 작은 해상도) 특징부터 꺼내서 결합
            x = torch.cat([x, h.pop()], dim=1) # 채널 결합 (Concat)
            x = block1(x, t)

            x = torch.cat([x, h.pop()], dim=1) # 채널 결합 (Concat)
            x = block2(x, t)

            x = attn(x)
            x = upsample(x)      # 해상도 복구 (H, W -> 2H, 2W)

        # [6] Final (원본 크기 복구 및 최종 노이즈 예측)
        x = torch.cat([x, r], dim=1)       # 최초의 특징(r)까지 결합
        x = self.final_res_block(x, t)
        return self.final_conv(x)          # (B, out_dim, H, W)


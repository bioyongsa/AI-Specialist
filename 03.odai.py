## 1. Prunning for CNN
### 1.1. Fine-grained Pruning
def prune_weight_fine_grained(weight: torch.Tensor, sparsity: float) -> None:  # 가중치 텐서에 대해 fine-grained pruning을 수행
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:      # 모든 가중치를 제거
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:    # 모든 가중치를 유지
        return torch.ones_like(weight)

    num_pruned_elements = round(weight.numel() * sparsity)    # [SG] 제거할 원소 갯수
    importance = torch.abs(weight)                            # [SG] 가중치의 중요도를 절대값으로
    threshold = torch.kthvalue(importance.flatten(), num_pruned_elements)[0]    # [SG] pruning threshold
    mask = importance > threshold                             # [SG] threshold보다 큰 값들은 유지(1), 작은 값들은 제거(0)
    weight.mul_(mask)                                         # [SG] mask 적용하여 pruning
    return mask

### 1.2. Vector-level Pruning
def prune_weight_vector_level(weight: torch.Tensor, sparsity: float) -> None:  # 벡터 단위로 가중치를 프루닝
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:      # 모든 가중치를 제거
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:    # 모든 가중치를 유지
        return torch.ones_like(weight)

    num_vectors = weight.shape[0] * weight.shape[1] * weight.shape[2]        # [SG] 제거할 벡터 갯수
    num_pruned_vectors = round(num_vectors * sparsity)
    importance = weight.abs().sum(dim=(3,), keepdim=True)                    # [SG] 벡터의 중요도를 절대값으로
    threshold = torch.kthvalue(importance.flatten(), num_pruned_vectors)[0]  # [SG] pruning threshold
    mask = importance > threshold                                            # [SG] threshold보다 큰 값들은 유지(1), 작은 값들은 제거(0)
    mask = mask.expand_as(weight)                                            # [SG] mask를 가중치와 동일한 크기로 확장
    weight.mul_(mask)                                                        # [SG] mask 적용하여 pruning
    return mask

### 1.3. Kernel-level Pruning
def prune_weight_kernel_level(weight: torch.Tensor, sparsity: float) -> None:  # 커널 단위로 가중치를 프루닝
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:
        return torch.ones_like(weight)

    num_kernels = weight.shape[0] * weight.shape[1]                          # [SG] 프루닝할 커널 수 계산
    num_pruned_kernels = round(num_kernels * sparsity)
    importance = weight.abs().sum(dim=(2, 3), keepdim=True)                  # [SG] 각 커널의 중요도를 절대값의 합으로 계산
    threshold = torch.kthvalue(importance.flatten(), num_pruned_kernels)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)
    weight.mul_(mask)
    return mask

### 1.4. Channel-level Pruning
def prune_weight_channel_level(weight: torch.Tensor, sparsity: float) -> None:  # 채널 단위 프루닝
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:
        return torch.ones_like(weight)

    num_channels = weight.shape[1]                                            # [SG] 프루닝할 채널 수 계산
    num_pruned_channels = round(num_channels * sparsity)
    importance = weight.abs().sum(dim=(0, 2, 3), keepdim=True)                # 각 채널의 중요도를 절댓값 합으로 계산 (출력 채널, 커널 높이, 커널 너비에 대해 합산)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_channels)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)
    weight.mul_(mask)
    return mask

## 1.5. Global Magnitude Pruning: 모델의 모든 가중치를 하나의 집합으로 간주하고, 가중치의 절대값을 기준으로 중요도 평가
##      전체 모델에서 가장 작은 가중치를 제거. 전체 모델에서 가중치 크기에 따라 pruning 수행
class FineGrainedPrunerV2:
    def __init__(self, model, sparsity, global_prune=False):
        self.masks = FineGrainedPrunerV2.prune(model, sparsity, global_prune)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity, global_prune):
        masks = dict()
        if global_prune:            
            parameters_to_prune = []
            for name, param in model.named_parameters():
                if param.dim() > 1:          # conv, fc 레이어만 프루닝
                    parameters_to_prune.append(param.view(-1))

            all_weights = torch.cat(parameters_to_prune)              # [SG] 모든 weight를 하나의 tensor로 결합
            num_elements = all_weights.numel()                        # 
            num_zeros = round(num_elements * sparsity)                #
            importance = torch.abs(all_weights)                       # 
            threshold = torch.kthvalue(importance, num_zeros)[0]      # [SG] all_weights를 대상으로 global threshold

            for name, param in model.named_parameters():
                if param.dim() > 1:
                    mask = torch.abs(param.data) > threshold
                    masks[name] = mask
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1: # we only prune conv and fc weights
                    masks[name] = prune_weight_fine_grained(param, sparsity)
        return masks

## 2. Quantization for CNN
### 2.1. Linear Quantization
def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    scaled_tensor = fp_tensor/scale
    rounded_tensor = torch.round(scaled_tensor)
    rounded_tensor = rounded_tensor.to(dtype)
    shifted_tensor = rounded_tensor + zero_point
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    return quantized_tensor

### 2.2. scale 및 zero point
def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()
    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    zero_point = round(quantized_min - fp_min/scale)

    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else: # convert from float to int using round()
        zero_point = round(zero_point)
    return scale, int(zero_point)

### 2.3. Quantized Fully Connected Layer
def quantized_linear(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale):
  ...
    weight_scale = weight_scale.flatten().view(1, -1)
    output = output.float()

    output = output*(input_scale * weight_scale / output_scale)
    output = output + output_zero_point
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output

### 2.4. Quantized Convolutional Layer
def quantized_conv2d(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale,
                     stride, padding, dilation, groups):
    ...
    weight_scale = weight_scale.flatten().view(1, -1, 1, 1)
    output = output.float()

    output = output*(input_scale * weight_scale / output_scale)
    output = output + output_zero_point
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output

### 2.5. K-means Quantization
def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    if codebook is None:
        n_clusters = 2**bitwidth
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    quantized_tensor = codebook.centroids[codebook.labels]
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook

### 2.6. Quantization Aware Training
def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
        codebook.centroids[k] = torch.mean(fp32_tensor[codebook.labels==k])

## 3. Knowledge Distillation
### 3.1. Baseline Training
def train(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

### 3.2. Knowledge Distillation 학습 함수
def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, 
    T,  # temperature, soft_target_loss_weight, ce_loss_weight):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            student_prob = nn.functional.softmax(student_logits / T, dim=-1)
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - student_prob.log())) / student_prob.size(0) * (T**2)
            label_loss = ce_loss(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

### 3.3. Cosine Similarity 기반 Knowledge Distillation 학습 함수
def train_cosine_loss(teacher, student, train_loader, epochs, learning_rate,
    hidden_rep_loss_weight, ce_loss_weight):
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)
            student_logits, student_hidden_representation = student(inputs)
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation,
                                          target=torch.ones(inputs.size(0)).cuda())
            label_loss = ce_loss(student_logits, labels)
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

### 3.4. MSE Loss 기반 Hint-based Knowledge Distillation 학습 함수
def train_mse_loss(teacher, student, train_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(train_loader))
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)
            student_logits, regressor_feature_map = student(inputs)
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)
            label_loss = ce_loss(student_logits, labels)
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

## 4. Prunning for LLM
### 4.1. Magnitude-based Prunning
def prune_magnitude(model, sparsity):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            num_elements = W.numel()
            num_zeros = round(num_elements * sparsity)
            importance = torch.abs(W)
            threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]
            mask = importance > threshold
            W.mul_(mask)

### 4.2. Calibration Dataset 준비
def get_calib_feat(model, tokenizer, samples):
    input_dict = dict()
    nsamples_dict = dict()
    def add_batch(m, x, y, name):
        if name not in input_dict:
            input_dict[name] = torch.zeros((m.weight.data.shape[1]), device=m.weight.data.device)
            nsamples_dict[name] = 0
        if isinstance(x, tuple):
            x = x[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        tmp = x.shape[0]
        if len(x.shape) == 3:
            x = x.reshape((-1, x.shape[-1]))
        x = x.t()
        input_dict[name] *= nsamples_dict[name] / (nsamples_dict[name] + tmp)
        nsamples_dict[name] += tmp
        x = x.type(torch.float32)

        activation_norm = torch.norm(x, p=2, dim=1) ** 2    # 중요!!!
        input_dict[name] += activation_norm / nsamples_dict[name]

### 4.3. Wanda Prunning (중요!!)
def prune_wanda(model, sparsity, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            row, col = W.shape
            num_zeros_per_row = round(col * sparsity)
            importance = torch.abs(W) * input_feat[n]
            threshold = torch.kthvalue(importance, num_zeros_per_row, dim=1)[0]
            mask = importance > threshold.reshape(row, 1)
            W.mul_(mask)

## 5. Quantization for LLM
### 5.1. Simulated Quantization
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    ...
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int       # scale = (α - β) / (2^n_bit - 1)
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)  # zero = round(-β / scale)
    ...    
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int) # 양자화(Quantization) 단계: 실수값을 정수 범위 [0, 2^n_bit - 1]로 매핑    
    w = (w - zeros) * scales                                     # 복원 단계(Dequantization): 정수값을 다시 실수 범위로 변환 (pseudo quantization)

    w = w.reshape(org_w_shape)                                   # 원래 텐서 형태로 복원
    return w

### 5.3. Scale 1% Salient Channels
def pseudo_quantize_model_weight_scaleup(
    model, w_bit, q_group_size, input_feat, scale_factor
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()
            num_samples = int(len(importance) * 0.01)
            outlier_mask = torch.topk(importance, int(len(importance) * 0.01))[1]
            assert outlier_mask.dim() == 1
            m.weight.data[:, outlier_mask] *= scale_factor
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)
            m.weight.data[:, outlier_mask] /= scale_factor

### 5.4. Scale Factor Search
def auto_scale_block(module, name, w_bit, q_group_size, input_feat):
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]
        s_x = x.view(-1, x.shape[-1]).abs().mean(0)
        s_x = torch.clamp(s_x, 1e-5)
        best_error = torch.inf
        best_ratio = -1
        best_scales = 0
        n_grid = 20
        history = []
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            scales = s_x ** ratio            # 공식
            assert scales.shape == s_x.shape
            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)
            for fc in linears2scale:
                scales = scales.to(fc.weight.device)                
                fc.weight.data *= scales    # Step 3: scale_factor를 이용해 중요한 가중치 채널의 값을 확대합니다.
                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)
                fc.weight.data /= scales    # Step 4: scale_factor를 이용해 중요한 가중치 채널의 값을 다시 축소합니다.
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # (1) Self-Attention 입력 부분 (Q, K, V projection)
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # (2) Self-Attention 출력 부분 (out_proj)
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.self_attn.out_proj, [module.self_attn.out_proj], inp)
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # (3) Feed-Forward Network 첫 번째 FC (fc1)
    inp = input_feat[name + '.fc1']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # (4) Feed-Forward Network 두 번째 FC (fc2)
    inp = input_feat[name + '.fc2']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    scale_fc_fc(module.fc1, module.fc2, final_scales)

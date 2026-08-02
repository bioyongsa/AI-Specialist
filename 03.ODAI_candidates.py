01.CNN_Prunning # ---------------------------------------------
#13
def prune_weight_fine_grained(weight: torch.Tensor, sparsity: float) -> None:
    sparsity = min(1.0, max(0.0, sparsity))

    # 특수한 경우 처리
    if sparsity == 1.0:  # 모든 가중치를 제거
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:  # 모든 가중치를 유지
        return torch.ones_like(weight)

    ##################### YOUR CODE STARTS HERE #####################
    num_pruned_elements = round(weight.numel() * sparsity)
    importance = torch.abs(weight)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_elements)[0]
    mask = importance > threshold
    ##################### YOUR CODE ENDS HERE #######################

    weight.mul_(mask)
    return mask

# 14
def prune_weight_vector_level(weight: torch.Tensor, sparsity: float) -> None:
    ...

    num_vectors = weight.shape[0] * weight.shape[1] * weight.shape[2]
    num_pruned_vectors = round(num_vectors * sparsity)
    importance = weight.abs().sum(dim=(3,), keepdim=True)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_vectors)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)

    weight.mul_(mask)
    return mask

# 15
def prune_weight_kernel_level(weight: torch.Tensor, sparsity: float) -> None:
    ...

    num_kernels = weight.shape[0] * weight.shape[1]
    num_pruned_kernels = round(num_kernels * sparsity)
    importance = weight.abs().sum(dim=(2, 3), keepdim=True)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_kernels)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)
    weight.mul_(mask)

    return mask

# 16
def prune_weight_channel_level(weight: torch.Tensor, sparsity: float) -> None:
    ...

    num_channels = weight.shape[1]
    num_pruned_channels = round(num_channels * sparsity)
    importance = weight.abs().sum(dim=(0, 2, 3), keepdim=True)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_channels)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)
    weight.mul_(mask)

    return mask

# 25
class FineGrainedPrunerV2:
    ...
    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity, global_prune):        
        ...
        if global_prune:
            ...
            ##################### YOUR CODE STARTS HERE #####################
            all_weights = torch.cat(parameters_to_prune)            
            num_elements = all_weights.numel()
            num_zeros = round(num_elements * sparsity)
            importance = torch.abs(all_weights)
            threshold = torch.kthvalue(importance, num_zeros)[0]
            ##################### YOUR CODE ENDS HERE #######################

            # threshold 기반 마스크 생성
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    mask = torch.abs(param.data) > threshold
                    masks[name] = mask
        else:
            ...        

02.CNN_Quantization # ---------------------------------------------
# 4
class VGG(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    layers = []
    counts = defaultdict(int)

    def add(name: str, layer: nn.Module) -> None:
      layers.append((f"{name}{counts[name]}", layer))
      counts[name] += 1

    in_channels = 3
    for x in self.ARCH:
      if x != 'M':
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))
    add("avgpool", nn.AvgPool2d(2))
    self.backbone = nn.Sequential(OrderedDict(layers))
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    # x = x.mean([2, 3])
    x = x.view(x.shape[0], -1)

    # classifier: [N, 512] => [N, 10]
    x = self.classifier(x)
    return x

  # 14
def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    ...

    ############### YOUR CODE STARTS HERE ###############
    scaled_tensor = fp_tensor/scale
    rounded_tensor = torch.round(scaled_tensor)
    ############### YOUR CODE ENDS HERE #################

    rounded_tensor = rounded_tensor.to(dtype)

    ############### YOUR CODE STARTS HERE ###############
    shifted_tensor = rounded_tensor + zero_point
    ############### YOUR CODE ENDS HERE #################

    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    return quantized_tensor

# 17
def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    ############### YOUR CODE STARTS HERE ###############
    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    zero_point = round(quantized_min - fp_min/scale)
    ############### YOUR CODE ENDS HERE #################

    ,,,

# 22
def linear_quantize_weight_per_channel(tensor, bitwidth):
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)

    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0

# 26
def quantized_linear(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale):
    ...

    weight_scale = weight_scale.flatten().view(1, -1)
    output = output.float()

    ############### YOUR CODE STARTS HERE ###############
    output = output*(input_scale * weight_scale / output_scale)
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE #################
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output

# 30
def quantized_conv2d(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale,
                     stride, padding, dilation, groups):
    ...
    
    weight_scale = weight_scale.flatten().view(1, -1, 1, 1)
    output = output.float()

    ############### YOUR CODE STARTS HERE ###############
    output = output*(input_scale * weight_scale / output_scale)
    output = output + output_zero_point
    ############### YOUR CODE ENDS HERE #################
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output

# 39
def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    ...
    if codebook is None:
        ############### YOUR CODE STARTS HERE ###############
        n_clusters = 2**bitwidth
        ############### YOUR CODE ENDS HERE #################
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### YOUR CODE STARTS HERE ###############
    quantized_tensor = codebook.centroids[codebook.labels]
    ############### YOUR CODE ENDS HERE #################
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook

# 44
def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    ...
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
    ############### YOUR CODE STARTS HERE ###############
        codebook.centroids[k] = torch.mean(fp32_tensor[codebook.labels==k])
    ############### YOUR CODE ENDS HERE #################

# 46
# 모델 정의
import torch.nn.functional as F
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        ...

        # 양자화 스텁 추가
        self.quant = torch.quantization.QuantStub()  # 입력을 양자화
        self.dequant = torch.quantization.DeQuantStub()  # 출력을 역양자화

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Pooling and Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.quant(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.contiguous().view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dequant(x)
        return x
    
03.Knowledge_Distillation # ---------------------------------------------
# 12
def train_knowledge_distillation(teacher,
                                 student,
                                 train_loader,
                                 epochs,
                                 learning_rate,
                                 T,  # temperature
                                 soft_target_loss_weight,
                                 ce_loss_weight):
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

            ##################### YOUR CODE STARTS HERE #####################
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            student_prob = nn.functional.softmax(student_logits / T, dim=-1)

            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - student_prob.log())) / student_prob.size(0) * (T**2)

            label_loss = ce_loss(student_logits, labels)
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
            ##################### YOUR CODE ENDS HERE #######################

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

# 18
def train_cosine_loss(teacher,
                      student,
                      train_loader,
                      epochs,
                      learning_rate,
                      hidden_rep_loss_weight,
                      ce_loss_weight):
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

            ##################### YOUR CODE STARTS HERE #####################
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(inputs)

            student_logits, student_hidden_representation = student(inputs)
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation,
                                          target=torch.ones(inputs.size(0)).cuda())

            label_loss = ce_loss(student_logits, labels)
            loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss
            ##################### YOUR CODE ENDS HERE #######################

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

# 24
def train_mse_loss(teacher,
                   student,
                   train_loader,
                   epochs,
                   learning_rate,
                   feature_map_weight,
                   ce_loss_weight):
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

            ##################### YOUR CODE STARTS HERE #####################
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            student_logits, regressor_feature_map = student(inputs)
            hidden_rep_loss = mse_loss(regressor_feature_map, teacher_feature_map)

            label_loss = ce_loss(student_logits, labels)
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss
            ##################### YOUR CODE ENDS HERE #######################

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

04.LLM_Prunning # ---------------------------------------------
# 4
@torch.no_grad()
def prune_magnitude(model, sparsity):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            ##################### YOUR CODE STARTS HERE #####################
            num_elements = W.numel()
            num_zeros = round(num_elements * sparsity)
            importance = torch.abs(W)
            threshold = torch.kthvalue(importance.flatten(), num_zeros)[0]
            mask = importance > threshold
            ##################### YOUR CODE ENDS HERE #######################
            W.mul_(mask)

# 6
def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    ...
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

samples = get_calib_dataset(tokenizer)

# 9
@torch.no_grad()
def prune_wanda(model, sparsity, input_feat):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in n:
            W = m.weight.data
            ##################### YOUR CODE STARTS HERE #####################
            row, col = W.shape
            num_zeros_per_row = round(col * sparsity)
            importance = torch.abs(W) * input_feat[n]
            threshold = torch.kthvalue(importance, num_zeros_per_row, dim=1)[0]
            mask = importance > threshold.reshape(row, 1)
            ##################### YOUR CODE ENDS HERE #######################
            W.mul_(mask)

05.LLM_Quantization # ---------------------------------------------
# 5
# 핵심 양자화 함수 (simulated quantization, 즉 실제 비트 축소 없이 효과만 모사)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    ...

    # 각 행(또는 그룹)별 최대값(α)과 최소값(β) 계산
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)

    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)

    # NaN 검증
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    w = (w - zeros) * scales
    ...

# 10
@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(
    model, w_bit, q_group_size, input_feat, scale_factor
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()
            num_samples = int(len(importance) * 0.01)

            ############### YOUR CODE STARTS HERE ###############
            outlier_mask = torch.topk(importance, int(len(importance) * 0.01))[1]
            ############### YOUR CODE ENDS HERE #################
            assert outlier_mask.dim() == 1

            m.weight.data[:, outlier_mask] *= scale_factor
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############
            m.weight.data[:, outlier_mask] /= scale_factor
            ############### YOUR CODE ENDS HERE #################

# 13
@torch.no_grad()
def auto_scale_block(module, name, w_bit,
                     q_group_size,
                     input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        ...

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)
        s_x = torch.clamp(s_x, 1e-5)

        ...
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############
            scales = s_x ** ratio
            ############### YOUR CODE ENDS HERE #################
            assert scales.shape == s_x.shape
            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:
                scales = scales.to(fc.weight.device)
                ############### YOUR CODE STARTS HERE ###############
                fc.weight.data *= scales
                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)
                fc.weight.data /= scales
                ############### YOUR CODE ENDS HERE #################

# 18
def smooth_ln_fcs_by_scale(ln, fcs, scale):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)

    ############### YOUR CODE STARTS HERE ###############
    ln.weight.div_(scale)
    ln.bias.div_(scale)
    ############### YOUR CODE ENDS HERE #################

    for fc in fcs:
        ############### YOUR CODE STARTS HERE ###############
        fc.weight.mul_(scale)
        ############### YOUR CODE ENDS HERE #################
        
# 20
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    ,,,
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (
        ############### YOUR CODE STARTS HERE ###############
        act_scales.pow(alpha) / weight_scales.pow(1 - alpha)
        ############### YOUR CODE ENDS HERE #################
    )

    scales.clamp(min=1e-5).to(device).to(dtype)
    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
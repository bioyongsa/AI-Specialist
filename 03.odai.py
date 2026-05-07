## Prunning

def prune_weight_fine_grained(weight: torch.Tensor, sparsity: float) -> None:  # 가중치 텐서에 대해 fine-grained pruning을 수행
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:  # 모든 가중치를 제거
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:  # 모든 가중치를 유지
        return torch.ones_like(weight)

    num_pruned_elements = round(weight.numel() * sparsity)
    importance = torch.abs(weight)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_elements)[0]
    mask = importance > threshold
    weight.mul_(mask)
    return mask

mask_fine_grained = prune_weight_fine_grained(weight.clone(), prune_sparsity)
draw_weight_distribution(mask_fine_grained, title="Fine-grained Pruning Mask")

def prune_weight_vector_level(weight: torch.Tensor, sparsity: float) -> None:  # 벡터 단위로 가중치를 프루닝
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:  # 모든 가중치를 제거
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:  # 모든 가중치를 유지
        return torch.ones_like(weight)

    num_vectors = weight.shape[0] * weight.shape[1] * weight.shape[2]
    num_pruned_vectors = round(num_vectors * sparsity)
    importance = weight.abs().sum(dim=(3,), keepdim=True)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_vectors)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)
    weight.mul_(mask)
    return mask

mask_vector_level = prune_weight_vector_level(weight.clone(), prune_sparsity)
draw_weight_distribution(mask_vector_level, title="Vector-level Pruning Mask")

def prune_weight_kernel_level(weight: torch.Tensor, sparsity: float) -> None:  # 커널 단위로 가중치를 프루닝
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:
        return torch.ones_like(weight)

    num_kernels = weight.shape[0] * weight.shape[1]
    num_pruned_kernels = round(num_kernels * sparsity)
    importance = weight.abs().sum(dim=(2, 3), keepdim=True)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_kernels)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)
    weight.mul_(mask)
    return mask

def prune_weight_channel_level(weight: torch.Tensor, sparsity: float) -> None:  # 채널 단위 프루닝
    sparsity = min(1.0, max(0.0, sparsity))
    if sparsity == 1.0:
        weight.zero_()
        return torch.zeros_like(weight)
    elif sparsity == 0.0:
        return torch.ones_like(weight)

    num_channels = weight.shape[1]
    num_pruned_channels = round(num_channels * sparsity)
    importance = weight.abs().sum(dim=(0, 2, 3), keepdim=True)
    threshold = torch.kthvalue(importance.flatten(), num_pruned_channels)[0]
    mask = importance > threshold
    mask = mask.expand_as(weight)
    weight.mul_(mask)
    return mask

mask_channel_level = prune_weight_channel_level(weight.clone(), prune_sparsity)
draw_weight_distribution(mask_channel_level, title="Channel-level Pruning Mask")

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
                if param.dim() > 1:  # conv, fc 레이어만 프루닝
                    parameters_to_prune.append(param.view(-1))

            all_weights = torch.cat(parameters_to_prune)
            num_elements = all_weights.numel()
            num_zeros = round(num_elements * sparsity)
            importance = torch.abs(all_weights)
            threshold = torch.kthvalue(importance, num_zeros)[0]

            for name, param in model.named_parameters():
                if param.dim() > 1:
                    mask = torch.abs(param.data) > threshold
                    masks[name] = mask
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1: # we only prune conv and fc weights
                    masks[name] = prune_weight_fine_grained(param, sparsity)
        return masks

mask_kernel_level = prune_weight_kernel_level(weight.clone(), prune_sparsity)
draw_weight_distribution(mask_kernel_level, title="Kernel-level Pruning Mask")

def get_sparsity_schedule(num_epochs: int,
                          epoch_start: int,
                          epoch_end: int,
                          sparsity_start: float,
                          sparsity_end: float,
                          exponent: int,
                          ) -> List[float]:
    sparsity_schedule = []
    
    for epoch in range(num_epochs):        
        if epoch < epoch_start:
            sparsity = 0
        elif epoch >= epoch_end:
            sparsity = sparsity_end
        else:
            sparsity = sparsity_end + (sparsity_start - sparsity_end) * (1 - (epoch - epoch_start) / (epoch_end - epoch_start)) ** exponent
        sparsity_schedule.append(sparsity)
    return sparsity_schedule

linear_sparsity_schedule = get_sparsity_schedule(num_finetune_epochs, 0, 2, 0.9, target_sparsity, 1)
cubic_sparsity_schedule = get_sparsity_schedule(num_finetune_epochs, 0, 2, 0.9, target_sparsity, 3)

## CNN Quantization
# 1. Linear Quantization
def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    scaled_tensor = fp_tensor/scale
    rounded_tensor = torch.round(scaled_tensor)
    rounded_tensor = rounded_tensor.to(dtype)
    shifted_tensor = rounded_tensor + zero_point
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    return quantized_tensor

# 2. scale 및 zero point
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

# 3. Quantized Fully Connected Layer
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

# 4. Quantized Convolutional Layer
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

# 5. K-means Quantization
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

# 6. Quantization Aware Training
def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
        codebook.centroids[k] = torch.mean(fp32_tensor[codebook.labels==k])

## Knowledge Distillation
# 1. Baseline Training
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

# 2. Knowledge Distillation 학습 함수
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

# 3. Cosine Similarity 기반 Knowledge Distillation 학습 함수
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

# 4. MSE Loss 기반 Hint-based Knowledge Distillation 학습 함수
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

### Prunning for LLM
# 1. Magnitude-based Prunning
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

# 2. Calibration DAtaset 준비
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

# 3. Wanda Prunning (중요!!)
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

tokenizer = tiktoken.get_encoding("gpt2")
tokenids = tokenizer.encode(text)
strings = tokenizer.decode(tokenids)

tokens = [tokenizer.decode([t]) for t in tokenids]	# tensor인 경우
input_tensor = torch.tensor([tokenids])


class GPTDataSetV1(Dataset):
	def __init__(self, ...):
		for i in range(0, len(token_ids) - max_length, stride):
			input_chunk = token_ids[i: i + max_length]
			target_chunk = token_ids[i + 1: i + 1 + max_length]
			self.input_ids.append(torch.tensor(input_chunk))
			self.target_ids.append(torch.tensor(target_chunk))

token_embedding_layer = torch.nn.Embedding(vocab_size, out_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, out_dim)
token_embeddings = token_embedding_layer(input_tensor)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
input_embeddings = token_embeddings + pos_embeddings

class CausalAttention(nn.Module):
	def __init__(self, .... ):
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		...
		self.dropout = nn.Dropout(dropout)
		x = torch.ones(context_length, context_length)
		self.register_buffer(
			'mask', torch.triu(x, diagonal=1))
	def forward(self, x):
		queries = self.W_query(x)
		...
		attn_scores = queries @ keys.transpose(1, 2)
		mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
		attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		attn_weights = self.dropout(attn_weights)
		context_vec = attn_weights @ values
		return context_vec

class MultiHeadAttention(nn.Module):
	def __init__(self, ....):
		super().__init__()
		self.d_out = d_out
		self.num_heads = num_heads
		self.head_dim = d_out  // num_heads
		self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
		...
		self.out_proj = nn.Linear(d_out, d_out)
		self.dropout = nn.Dropout(dropout)
		x = torch.ones(context_length, context_length)
		self.register_buffer(
			'mask', torch.triu(x, diagonal=1))
	def forward(self, x):
		b, num_tokens, d_in = x.shape
		keys = self.W_key(x)
		...
		keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
		...
		keys = keys.transpose(1, 2)
		... 
		attn_scores = queries @ keys.transpose(2, 3)
		mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
		attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		attn_weights = self.dropout(attn_weights)
		context_vec = (attn_weights @ values).transpose(1, 2)
		context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
		context_vec = self.out_proj(context_vec)
		return context_vec

def make_hook():
	def hook_fn(module, input, output):
		x = input[0]
		all_attentions = []
		b, num_tokens, _ = x.shape
		keys = module.W_key(x)
		...
		keys = keys.view(b, num_tokens, module.num_heads, module.head_dim)
		...
		keys = keys.transpose(1, 2)
		...
		attn_scores = queries @ keys.transpose(2, 3)
		mask_bool = module.mask.bool()[:num_tokens, :num_tokens]
		attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
		attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
		all_attentions.append(attn_weights.detach())

class LayerNorm(nn.Module):
	def __init__(self, emb_dim):
		super().__init__()
		self.eps = 1e-5
		self.scale = nn.Parameter(torch.ones(emb_dim))
		self.shift = nn.Parameter(torch.zeros(emb_dim))
	def forward(self, x):
		mean = x.mean(dim=-1, keepdim=True)
		var = x.var(dim=-1, keepdim=True, unbiased=False)
		norm_x = (x - mean) / torch.sqrt(var + self.eps)
		return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
			nn.GELU(),
			nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
		)
	def forward(self, x):
		return self.layers(x)

class TransformerBlock(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.att = MultiHeadAttention(
			d_in = cfg["emb_dim"],
			d_out = cfg["emb_dim"],
			context_length = cfg["context_length"],
			num_heads = cfg["n_heads"],
			dropout = cfg["drop_rate"],
			qkv_bias = cfg["qkv_bias"]
		)
		self.ff = FeedForward(cfg)
		self.norm1 = LayerNorm(cfg["emb_dim"])
		self.norm2 = LayerNorm(cfg["emb_dim"])
		self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

	def forward(self, x):
		shortcut = x
		x = self.norm1(x)
		x = self.att(x)
		x = self.drop_shortcut(x)
		x = x + shortcut
		shortcut = x
		x = self.norm2(x)
		x = self.ff(x)
		x = self.drop_shortcut(x)
		x = x + shortcut
		return x

class GPTModel(nn.Module):
	def __init__(self, cfg):
		super().__init__()
		self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
		self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
		self.drop_emb = nn.Dropout(cfg["drop_rate"])
		self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
		self.final_norm = LayerNorm(cfg["emb_dim"])
		self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
	def forward(self, in_idx):
		batch_size, seq_len = in_idx.shape
		tok_embeds = self.tok_emb(in_idx)
		pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
		x = tok_embeds + pos_embeds
		x = self.drop_emb(x)
		x = self.trf_blocks(x)
		x = self.final_norm(x)
		logits = self.out_head(x)
		return logits

def generate_text_simple(model, idx, max_new_tokens, context_size):
	for _ in range(max_new_tokens):
		idx_cond = idx[:, -context_size:]
		with torch.no_grad():
			logits = model(idx_cond)
		logits = logits[:, -1, :]
		idx_next = torch.argmax(logits, dim=-1, keepdim=True)
		idx = torch.cat((idx, idx_next), dim=1)
	return idx

def calc_loss_batch(input_batch, target_batch, model, device):
	input_batch, target_batch = input_batch.to(device), target_batch.to(device)
	logits = model(input_batch)
	loss = torch.nn.functional.cross_entropy(
		logits.flatten(0, 1),
		target_batch.flatten()
	)
	return loss

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model_simple(...):
	train_losses, val_losses, track_tokens_seen = [], [], []
	tokens_seen = 0
	global_step = -1
	for epoch in range(num_epoch):
		model.train()
		for input_batch, target_batch in train_loader:
			optimizer.zero_grad()
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			loss.backward()
			optimizer.step()
			tokens_seen += input_batch.numel()
			global_step += 1
			
			if global_step % eval_freq == 0:
				train_loss, val_loss = evaluate_model( .... )
				train_losses.append(train_loss)
				val_losses.append(val_loss)
				track_tokens_seen.append(tokens_seen)
		generate_and_print_sample( ... )
	return train_losses, val_losses, track_tokens_seen

def text_to_token_ids(text, tokenizer):
	encoded = tokenizer.encode(text)
	encoded_tensor = torch.tensor(encoded).unsqueeze(0)
	return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
	flat = token_ids.squeeze(0)
	return tokenizer.decode(flat.tolist())

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
	model.eval()
	correct_predictions, num_examples = 0, 0

	if num_batches is None:
		num_batches = len(data_loader)
	else:
		num_batches = min(num_batches, len(data_loader))

	for i, (input_batch, target_batch) in enumerate(data_loader):
		if i < num_batches:
			input_batch, target_batch = input_batch.to(device), target_batch.to(device)
			with torch.no_grad():
				logits = model(input_batch)[:, -1, :]
			predicted_labels = torch.argmax(logits, dim=-1)
			num_examples += predicted_labels.shape[0]
			correct_predictions += (predicted_labels == target_batch).sum().item()
		else:
			break
	return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
	input_batch, target_batch = input_batch.to(device), target_batch.to(device)
	logits = model(input_batch)[:, -1, :]
	loss = torch.nn.functional.cross_entropy(logits, target_batch)
	return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
	total_loss = 0.
	if len(data_loader) == 0:
		return float("nan")
	elif num_batches is None:
		num_batches = len(data_loader)
	else:
		num_batches = min(num_batches, len(data_loader))

	for i, (input_batch, target_batch) in enumerate(data_loader):
		if i < num_batches:
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			total_loss += loss.item()
		else:
			break
	return total_loss / num_batches

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
	train_losses, val_losses, train_accs, val_accs = [], [], [], []
	examples_seen, global_step = 0, -1
	for epoch in range(num_epochs):
		model.train()
		for input_batch, target_batch in train_loader:
			optimizer.zero_grad()
			loss = calc_loss_batch(input_batch, target_batch, model, device)
			loss.backward()
			optimizer.step()
			examples_seen += input_batch.shape[0]
			global_step += 1

			if global_step % eval_freq == 0:
				train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
				train_losses.append(train_loss)
				val_losses.append(val_loss)
		train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
		val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
		train_accs.append(train_accuracy)
		val_accs.append(val_accuracy)

class LoRALayer(nn.Module):
	def __init__(self, in_dim, out_dim, rank, alpha):
		super().__init__()
		self.A = nn.Parameter(torch.empty(in_dim, rank))
		nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
		self.B = nn.Parameter(torch.zeros(rank, out_dim))
		self.alpha, self.rank = alpha, rank
	def forward(self, x):
		x = (self.alpha / self.rank) * (x @ self.A @ self.B)
		return x

class LinearWithLoRA(nn.Module):
	def __init__(self, linear, rank, alpha):
		super().__init__()
		self.linear = linear
		self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
	def forward(self, x):
		return self.linear(x) + self.lora(x)

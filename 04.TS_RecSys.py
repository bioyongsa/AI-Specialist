01.ts-solution # -----------------------------------------------------------------------------
# 10
input_size = X_train.shape[-1]
num_layers = 2
hidden_size = 64

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # (lstm): LSTM(1, 64, num_layers=2, batch_first=True)

        self.linear = nn.Linear(hidden_size, 1)
        # (linear): Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)

model = LSTMModel(input_size, hidden_size, num_layers).to(device)
print(model)

# 11
def train(model, train_loader, test_loader):
  train_hist = []
  test_hist = []
  num_epochs = 10
  loss_fn = nn.MSELoss(reduction='mean')
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(num_epochs):
    total_train_loss = 0.0
    total_test_loss = 0.0

    # train
    model.train()
    for batch_x, batch_y in train_loader:
      batch_x, batch_y = batch_x.to(device), batch_y.to(device)
      pred = model(batch_x)[:, -1, 0]
      loss = loss_fn(pred, batch_y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_train_loss += loss.item()

    avg_loss = total_train_loss / len(train_loader)
    train_hist.append(avg_loss)

    # evaluate
    model.eval()
    with torch.no_grad():
      for test_x, test_y in test_loader:
        test_x, test_y = test_x.to(device), test_y.to(device)
        test_pred = model(test_x)[:, -1, 0]
        test_loss = loss_fn(test_pred, test_y)
        total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_hist.append(avg_test_loss)

02.RecSys_GCF # -----------------------------------------------------------------------------
# 9
### TODO : NGCF Layer 완성 ###

class NGCFLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(input_dim, output_dim)
        self.W2 = nn.Linear(input_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, edge_index, node_features, user_num, item_num):
        '''
        edge_index : 엣지 정보 (src, dst)의 집합.
        node_features: node별 이전 layer에서 생성된 벡터 정보가 담긴 matrix (H^(l-1)) (|V| * d)
        '''

        src, dst = edge_index # src : user, dst : movie

        # calculate node degree
        deg = torch.zeros(node_features.size(0), device=node_features.device)
        # calculate user degree
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
        # calculate movie degree
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

        # calculate 1/(root(deg(u)) * root(deg(i))) for all edge
        norm = 1.0/torch.sqrt(deg[src]*deg[dst])

        src_feat = node_features[src] # H_u
        dst_feat = node_features[dst] # H_i

        edge_messages_for_src = self.W1(dst_feat) + self.W2(dst_feat*src_feat)
        edge_messages_for_src *= norm.unsqueeze(1)

        edge_messages_for_dst = self.W1(src_feat) + self.W2(src_feat*dst_feat)
        edge_messages_for_dst *= norm.unsqueeze(1)

        aggregated_messages = torch.zeros_like(node_features)
        aggregated_messages.index_add(0, src, edge_messages_for_src)
        aggregated_messages[:user_num] += self.W1(node_features[:user_num])

        aggregated_messages.index_add_(0, dst, edge_messages_for_dst)
        aggregated_messages[user_num:] += self.W1(node_features[user_num:])

        aggregated_features = self.leaky_relu(aggregated_messages)

        # engineering approach
        # aggregated_features = self.dropout(aggregated_features)
        return aggregated_features

# 10
### TODO: NGCF 모델 완성.

class NGCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layer_dims, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(self.num_users+self.num_items,self.embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        self.layers = nn.ModuleList([
            NGCFLayer(input_dim=(embedding_dim if i == 0 else layer_dims[i - 1]),
                      output_dim=layer_dims[i], dropout=dropout)
            for i in range(len(layer_dims))
        ])

    def forward(self, edge_index):
        node_features = self.node_embeddings.weight
        layer_outputs = [node_features]
        for layer in self.layers:
            node_features = layer(edge_index, node_features,self.num_users, self.num_items)
            layer_outputs.append(node_features)

        # Hint: NGCF의 final feature(representation)은 layer 별 feature에 대한 concatenated vector
        # Hint: 최종 final feature matrix에는 [feuture_vector for users + feature_vector for items]가 들어있음.

        final_features = torch.concat(layer_outputs,dim=-1)
        user_features = final_features[:self.num_users]
        item_features = final_features[self.num_users:]
        return user_features, item_features

    def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb, reg_weight=1e-4):
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = reg_weight * (user_emb.norm(2).pow(2) + pos_item_emb.norm(2).pow(2) \
                                 + neg_item_emb.norm(2).pow(2)) / user_emb.size(0)
        return loss + reg_loss
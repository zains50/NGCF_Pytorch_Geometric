
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.utils import degree, to_networkx
import matplotlib.pyplot as plt
import torch.nn.init as init
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from bprloss import  compute_bpr_loss
from recmodel import RecSysGNN

import torch
from torch_geometric.utils import negative_sampling




device = "cuda"
data = torch.load("gow/gowalla_combined_all_64_het.pt").to(device)
train_data = torch.load("gow/gowalla_train_all_64_het.pt").to(device)
test_data = torch.load("gow/gowalla_test_all_64_het.pt").to(device)

num_users = 29858
num_items = 40981
num_nodes = num_users+num_items


edge_index = data.edge_index
homogeneous_edge_index = data.edge_index[[1,0]]
data.edge_index = torch.cat([edge_index, homogeneous_edge_index], dim=1)

# Convert back to PyTorch
all_edge_index = data.edge_index
test_edge_index = test_data.edge_index
train_edge_index  = train_data.edge_index


model = RecSysGNN(64,4,num_users,num_items,"NGCF").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    batch_size = 4096
    perm = torch.randperm(train_edge_index.size(1))
    train_edge_index_1 = train_edge_index[:, perm]
    positive_edges = train_edge_index_1

    for start in tqdm(range(0, num_users, batch_size)):
        end = min(start + batch_size, num_users)
        pos_users, pos_items = positive_edges[0,start:end], positive_edges[1,start:end]
        neg_items = torch.randint(num_users, num_users + num_items, (pos_items.size(0),))
        users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0 = model.encode_minibatch(pos_users,pos_items,neg_items,data.edge_index)

        # def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        # def compute_bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0,  pos_emb0, neg_emb0):

        bpr_loss, reg_loss = compute_bpr_loss(
            pos_users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0
        )
        DECAY = 0.0001
        reg_loss = DECAY * reg_loss
        final_loss = bpr_loss + reg_loss
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

    return final_loss

batch_size=1024
@torch.no_grad()
def test(k: int):
    precision = recall = total_examples = 0

    print("Test batches")
    for start in tqdm(range(0, num_users, batch_size)):
        end = min(start + batch_size, num_users)

        user_emb = model.embedding.weight[start:end]
        item_emb = model.embedding.weight[num_users:]  # all items

        logits = user_emb @ item_emb.t()  # [batch_size, num_items]

        mask = ((train_edge_index[0] >= start) &
                (train_edge_index[0] < end))
        logits[train_edge_index[0, mask] - start,
               train_edge_index[1, mask] - num_users] = float('-inf')

        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((test_edge_index[0] >= start) &
                (test_edge_index[0] < end))

        rows = test_edge_index[0, mask] - start
        cols = test_edge_index[1, mask] - num_users

        valid = (rows >= 0) & (rows < ground_truth.size(0)) & (cols >= 0) & (cols < ground_truth.size(1))
        rows = rows[valid]
        cols = cols[valid]

        ground_truth[rows, cols] = True
        node_count = degree(test_edge_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples




loss_list = []
percision_list = []
recall_20_list = []
abov2k_list = []
below2k_list = []
roc_list = []

for epoch in range(1, 1000):
    loss = train()
    precision, recall = test(k=20)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
          f'{precision:.4f}, Recall@20: {recall:.4f}')




print(loss_list)
print(percision_list)
print(recall_20_list)
print(abov2k_list)
print(below2k_list)
print(roc_list)



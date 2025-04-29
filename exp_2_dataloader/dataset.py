import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


class VTKG(Dataset):
    def __init__(self, data, max_vis_len):
        super().__init__()

        self.dir = f'data/{data}'
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []

        with open(os.path.join(self.dir, 'entities.txt'), 'r') as f:
            lines = f.readlines()
            for _, line in enumerate(lines):
                self.ent2id[line.strip()] = _
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(os.path.join(self.dir, 'relations.txt'), 'r') as f:
            lines = f.readlines()
            for _, line in enumerate(lines):
                self.rel2id[line.strip()] = _
                self.id2rel.append(line.strip())
        self.num_rel = len(self.rel2id)

        self.train = []
        with open(os.path.join(self.dir, 'train.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(os.path.join(self.dir, 'valid.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(os.path.join(self.dir, 'test.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                h, r, t = line.strip().split('\t')
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.filter_dict = {}
        for data_filter in [self.train, self.valid, self.test]:
            for triple in data_filter:
                h, r, t = triple
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, r, t)] = []
                self.filter_dict[(-1, r, t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

        self.max_vis_len_ent = max_vis_len
        self.max_vis_len_rel = max_vis_len

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        # h, r, t = self.train[idx]
        # if random.random() < 0.5:
        #     masked_triple = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
        #     label = h
        # else:
        #     masked_triple = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
        #     label = t
        return self.train[idx]


def vtkg_collate_fn(batch, num_ent, num_rel):
    batch_size = len(batch)
    inputs = []
    labels = []
    half = batch_size // 2
    for i in range(batch_size):
        h, r, t = batch[i]
        if i < half:
            masked_triple = [num_ent + num_rel, r + num_ent, t + num_rel]
            label = h
        else:
            masked_triple = [h + num_rel, r + num_ent, num_ent + num_rel]
            label = t
        inputs.append(masked_triple)
        labels.append(label)
    return torch.tensor(inputs), torch.tensor(labels)


if __name__ == '__main__':
    kg = VTKG('DB15K', -1)
    dataloader = DataLoader(dataset=kg,
                            batch_size=256,
                            collate_fn=lambda batch: vtkg_collate_fn(batch, kg.num_ent, kg.num_rel),
                            shuffle=True)
    for i, data in enumerate(dataloader):
        print(data[0][:128])
        print(data[0][128:])
        if i == 0:
            break

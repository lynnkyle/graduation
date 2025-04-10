# import json
# from collections import defaultdict, Counter
#
# import torch
#
#
# def load_ent_map(dataset):
#     ent_map = {}
#     if dataset == 'FB15K-237' or dataset == 'DB15K' or dataset == 'WN9':
#         with open(f'data/{dataset}/entities.txt', 'r') as f:
#             lines = f.readlines()
#             for step, line in enumerate(lines):
#                 ent_map[line.strip()] = int(step)
#     else:
#         with open(f'data/{dataset}/entity2id.txt', 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 _, id = line.strip().split(' ')
#                 ent_map[_] = int(id)
#     return ent_map
#
#
# """
#     一个实体对应多个token; 一个token对应一个特征向量
# """
#
#
# def get_entity_visual_tokens(dataset, max_num, type='beit'):
#     """
#         每个实体取最相关的前八个视觉token
#     """
#     if type == 'beit':
#         tokenized_result = json.load(open(f'tokens/{dataset}-visual.json', 'r'))
#         token_size = 8192  # bert的词汇表大小
#     elif type == 'vggan':
#         tokenized_result = json.load(open(f'tokens/{dataset}-visual-vqgan.json', 'r'))
#         token_size = 1024  # v
#     else:
#         raise NotImplementedError
#
#     """
#         构建tkn2ent字典: 记录每个token相关的entity
#         构建ent2tkn字典: 记录每个entity相关的token
#     """
#
#     tkn2ent = defaultdict(list)
#     for i in range(token_size + 1):
#         tkn2ent[i] = []
#     for entity in tokenized_result:
#         tkn_count = Counter(tokenized_result[entity])
#         selected_tkn = tkn_count.most_common(max_num)
#         for tkn, _ in selected_tkn:
#             tkn2ent[tkn].append(entity)
#
#     ent2id = load_ent_map(dataset)
#     tkn_id = list(tkn2ent.keys())
#     ent_id2tkn = defaultdict(list)
#     for i in range(len(tkn_id)):
#         for entity in tkn2ent[tkn_id[i]]:
#             ent_id2tkn[ent2id[entity]].append(i)
#
#     ent_id2tkns = []
#     ent_key_mask = []
#     for i in range(len(ent2id)):
#         if ent_id2tkn[i] != []:
#             ent_id2tkns.append(ent_id2tkn[i])
#             ent_key_mask.append(([False] * max_num))
#         else:
#             ent_id2tkns.append([token_size - 1] * max_num)
#             ent_key_mask.append(([True] * max_num))
#     return torch.LongTensor(ent_id2tkns).cuda(), torch.BoolTensor(ent_key_mask).cuda()
#
#
# def get_entity_textual_tokens(dataset, max_num, type='bert'):
#     """
#         每个实体取最相关的前四个文本token
#     """
#     # if dataset == 'DB15K':
#     #     return get_entity_textual_tokens_db15k(dataset, max_num, type)
#     if type == 'bert':
#         tokenized_result = json.load(open(f'tokens/{dataset}-textual.json', 'r'))
#         token_size = 30522
#     else:
#         raise NotImplementedError
#     tkn2ent = defaultdict(list)
#     for i in range(token_size + 1):
#         tkn2ent[i] = []
#     for entity in tokenized_result:
#         tkn_count = Counter(tokenized_result[entity])
#         selected_tkn = tkn_count.most_common(max_num)
#         for tkn, _ in selected_tkn:
#             tkn2ent[tkn].append(entity)
#
#     ent2id = load_ent_map(dataset)
#     tkn_id = list(tkn2ent.keys())
#     ent_id2tkn = defaultdict(list)
#     for i in range(len(tkn_id)):
#         for entity in tkn2ent[tkn_id[i]]:
#             ent_id2tkn[ent2id[entity]].append(i)
#
#     ent_id2tkns = []
#     ent_key_mask = []
#     for i in range(len(ent2id)):
#         if len(ent_id2tkn[i]) == max_num:
#             ent_id2tkns.append(ent_id2tkn[i])
#             ent_key_mask.append(([False] * max_num))
#         else:
#             s = ent_id2tkn[i]
#             ent_id2tkns.append(s + [14999] * (max_num - len(s)))
#             ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
#     return torch.LongTensor(ent_id2tkns).cuda(), torch.BoolTensor(ent_key_mask).cuda()
import json
import torch
from collections import defaultdict, Counter


def load_ent_map(dataset):
    ent_map = {}
    if dataset == "FB15K-237" or dataset == "DB15K" or dataset == "WN9":
        f = open("data/{}/entities.txt".format(dataset), "r")
        for line in f.readlines():
            ent = line.replace('\n', '')
            ent_map[ent] = ent
    else:
        f = open("data/{}/entity2id.txt".format(dataset), "r")
        for line in f.readlines():
            ent, id = line[:-1].split(' ')
            ent_map[ent] = int(id)
    return ent_map


def get_entity_visual_tokens(dataset, max_num, type="beit"):
    if type == "beit":
        tokenized_result = json.load(open("tokens/{}-visual.json".format(dataset), "r"))
        token_size = 8192
    elif type == 'vqgan':
        tokenized_result = json.load(open("tokens/{}-visual-vqgan.json".format(dataset), "r"))
        token_size = 1024
    else:
        raise NotImplementedError
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(token_size + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)

    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    # print(token_ids)
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    # json.dump(entity_to_token, open("{}-tokens-{}.json".format(dataset, max_num), "w"))
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        key = str(i) if dataset == "DB15K" else i
        if entity_to_token[key] != []:
            entid_tokens.append(entity_to_token[key])
            ent_key_mask.append(([False] * max_num))
        else:
            entid_tokens.append([token_size - 1] * max_num)
            ent_key_mask.append(([True] * max_num))
    return torch.LongTensor(entid_tokens).cuda(), torch.BoolTensor(ent_key_mask).cuda()


def get_entity_textual_tokens(dataset, max_num, type="bert"):
    tokenized_result = json.load(open("tokens/{}-textual.json".format(dataset), "r"))
    token_dict = defaultdict(list)
    entity_dict = load_ent_map(dataset)
    for i in range(30522 + 1):
        token_dict[i] = []
    for entity in tokenized_result:
        token_count = Counter(tokenized_result[entity])
        selected_tokens = token_count.most_common(max_num)
        for (token, num) in selected_tokens:
            token_dict[token].append(entity)
    num_count = [(k, len(token_dict[k])) for k in token_dict]
    num_count = sorted(num_count, key=lambda x: -x[1])
    token_ids = list(token_dict.keys())
    entity_to_token = defaultdict(list)
    for i in range(len(token_ids)):
        for ent in token_dict[token_ids[i]]:
            entity_to_token[entity_dict[ent]].append(i)
    entid_tokens = []
    ent_key_mask = []
    for i in range(len(entity_dict)):
        if entity_to_token[i] == max_num:
            entid_tokens.append(entity_to_token[i])
            ent_key_mask.append(([False] * max_num))
        else:
            s = entity_to_token[i]
            entid_tokens.append(s + [14999] * (max_num - len(s)))
            ent_key_mask.append(([False] * len(s) + [True] * (max_num - len(s))))
    return torch.LongTensor(entid_tokens).cuda(), torch.BoolTensor(ent_key_mask).cuda()


if __name__ == '__main__':
    visual_token = get_entity_visual_tokens('MKG-W', max_num=8)
    print(visual_token)
    textual_token = get_entity_textual_tokens('MKG-W', max_num=8)
    print(textual_token)

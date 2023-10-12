import torch
import numpy as np
from itertools import chain

np.random.seed(0)

def turn_list(s):
    if type(s) == list:
        return s
    elif type(s) == int:
        return [s]
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]*1.0

def construct_dict_sampleshapley(slist, cIdx, a_len):
    """Construct the position dict of sample shapley"""
    d = len(slist)
    cur = turn_list(slist[cIdx])  # current coalition
    sd = len(cur)

    positions_dict = {(i, fill):[] for i in range(sd+1) for fill in [0, 1]}
    # sample m times
    m = 100
    for cnt in range(m):
        perm = np.random.permutation(d)
        preO = []
        for idx in perm:
            if idx != cIdx:
                preO.append(turn_list(slist[idx]))
            else:
                break
        preO_list = list(chain.from_iterable(preO))
        pos_excluded = np.sum(to_categorical(preO_list, num_classes=a_len), axis=0)
        pos_included = pos_excluded + np.sum(to_categorical(turn_list(slist[cIdx]), num_classes=a_len), axis=0)
        positions_dict[(0, 0)].append(pos_excluded)
        positions_dict[(0, 1)].append(pos_included)
        for j in range(sd):
            subperm = np.random.permutation(d)
            subpreO = []
            for sidx in subperm:
                if sidx != cIdx:
                    subpreO.append(turn_list(slist[sidx]))
                else:
                    break

            tmp = cur[j]  # the elems in set S
            subpreO_list = list(chain.from_iterable(subpreO))
            pos_exc = np.sum(to_categorical(subpreO_list, num_classes=a_len), axis=0)
            pos_inc = pos_exc + np.sum(to_categorical(turn_list(tmp), num_classes=a_len), axis=0)
            positions_dict[(j + 1, 0)].append(pos_exc)
            positions_dict[(j + 1, 1)].append(pos_inc)

    keys, values = positions_dict.keys(), positions_dict.values()
    values = [np.array(value) for value in values]
    positions = np.concatenate(values, axis=0)

    key_to_idx = {}
    count = 0
    for i, key in enumerate(keys):
        key_to_idx[key] = list(range(count, count + len(values[i])))
        count += len(values[i])

    return positions_dict, key_to_idx, positions

def explain_shapley(predict, d, x, batch_all, key_to_idx):
    f_logits = []
    with torch.no_grad():
        logits = predict(x).cpu().numpy()                             # tensor to numpy
        for i in range(len(batch_all)):
            f_logits_l = predict(batch_all[i]).cpu().numpy()          # tensor to numpy
            f_logits.append(f_logits_l)
        f_logits = np.array(f_logits)
    prob_temp_shuffle_norm = (np.exp(logits) / np.sum(np.exp(logits), axis=1))
    discrete_probs = np.eye(len(logits[0]))[np.argmax(logits, axis=-1)]
    vals = np.sum(discrete_probs * f_logits, axis=1)

    # key_to_idx[key]: list of indices in original position.
    key_to_val = {key: np.array([vals[idx] for idx in key_to_idx[key]]) for key in key_to_idx}

    # Compute importance scores.
    phis = np.zeros(d)
    for i in range(d):
        phis[i] = np.mean(key_to_val[(i, 1)] - key_to_val[(i, 0)])

    return phis,logits

def compute_scores(slist, cIdx, feature, a_len, predict, args, device):
    positions_dict, key_to_idx, positions = construct_dict_sampleshapley(slist, cIdx, a_len)

    """ feature convert """
    feature_text = feature.cpu().numpy()                         # batch.sentence
    real_ids = []
    for f in feature_text:
        real_ids.append(f[0]) 

    """ padding """
    if a_len < args.minseqlen:
        batchtempt_x = torch.ones(args.minseqlen, 1, dtype=torch.int64)       # 1 -> <pad>
        batchtempt_x[0:a_len] = feature
        x = batchtempt_x.to(device)
    else:
        x = feature

    inputs = np.array(real_ids) * positions

    batch_all  = []   
    for j in range(inputs.shape[0]):                                                                         
        if a_len < args.minseqlen:
            batchtempt = torch.ones(args.minseqlen, 1, dtype=torch.int64)       # 1 -> <pad>
            input_d = []
            for x_node in inputs[j]:
                input_d.append([x_node])
            batchtempt[0:a_len] = torch.from_numpy(np.array(input_d)).long()
            batch_all.append(batchtempt.to(device))
        else:
            input_d = []
            for x_node in inputs[j]:
                input_d.append([x_node])
            batch_all.append(torch.from_numpy(np.array(input_d)).to(device).long())
    d = len(turn_list(slist[cIdx])) + 1

    shapvals, logits = explain_shapley(predict, d, x, batch_all, key_to_idx)

    return shapvals, logits





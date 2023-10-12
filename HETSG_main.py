import numpy as np
import argparse
import os
import time
import itertools
import random
import torch
import re
from HETSG_shapley import compute_scores, turn_list
from load_data import DATA
import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--task_name", default='sst-2', type=str, help="datasets")
parser.add_argument('--visualize', type=int, default=-1, help='index of the sentence to visualize, set to -1 to generate interpretations for all the sentences')
parser.add_argument('--start_pos', default=0, type=int, help='start position in the dataset')
parser.add_argument('--end_pos', default=-1, type=int, help='end position in the dataset')

parser.add_argument('--minseqlen', type=float, default=5, help='minimum sequence length')
parser.add_argument("--batch_size", default=1, type=int, help="batch_size")
parser.add_argument('--seed', type=int, default=111, help="random seed for initialization")
parser.add_argument('--gpu', default=0, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--n_gpu', type=int, default=1, help='the number of the used gpu')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if args.gpu > -1:
    args.device = "cuda"
else:
    args.device = "cpu"
device = torch.device(args.device)

def find_span(s):
    index_set = []
    for i in range(len(s)):
        if s[i] == 1:
            start = i
            for j in range(i+1,len(s)):
                if s[j] == 0:
                    end = j-1
                    break
                else:
                    end = len(s)-1
            index_set.append([start,end])
    return index_set

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
# Set seed
set_seed(args)

def tokenizer(s):                       # token
    s_clean = string_clean(s)
    return s_clean.split()

# Preprocessing
def string_clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?.\'\`]", " ", string)
    #string = re.sub(r"[^A-Za-z0-9\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string

def explain(args, model, data_iter, start_pos, end_pos, fileobject, vis=-1):

    if vis > -1:
        sen_iter = itertools.islice(data_iter, vis, vis+1)         
    else:
        if end_pos == -1:
            sen_iter = itertools.islice(data_iter, start_pos, None)
        else:
            sen_iter = itertools.islice(data_iter, start_pos, end_pos)

    count = 0
    acc = 0

    start_time = time.time()
    for batch in sen_iter:

        iter_time = time.time()
        count += 1

        fileobject.write(str(count))
        fileobject.write('\n')

        model.eval()
        feature = batch.sentence
        a_len = batch.sentence.size(0)
        pre_slist = list(range(a_len))
        pre_slen = len(pre_slist)
        output_tree = list(range(a_len))
        important_sen = []
        hier_tree = {}

        """ Write the source text  """
        for btxt in feature:
            if (wordvocab[btxt] != '<pad>' and wordvocab[btxt] != '<unk>'):
                fileobject.write(wordvocab[btxt])
                fileobject.write(' ')
        fileobject.write(' >> ')
        if str(batch.label.cpu().numpy()) == '[0]':
            fileobject.write('0')
            fileobject.write(' ||| ')
        else:
            fileobject.write('1')
            fileobject.write(' ||| ')

        """ compute the scores of the text span """
        for h in range(a_len - 1):
            totcombs = []
            ratios = []
            important_sub = []
            stn = {}
            tot_values = {}
            hier_tree[h] = []

            pre_slen = len(pre_slist)

            for k in range(pre_slen):
                scores, logits = compute_scores(pre_slist, k, feature,
                                        a_len, model, args, device)
                tot_values[k] = scores[0:1]/len(turn_list(pre_slist[k]))

                if type(pre_slist[k]) == int:
                    tempZ = [pre_slist[k]]
                else:
                    tempZ = pre_slist[k]
                hier_tree[h].append((tempZ,float(scores[0:1])))

            tot_values = sorted(tot_values.items(), key=lambda x: x[1], reverse=True)

            j = tot_values[0][0]
            if j == 0:
                coal = turn_list(pre_slist[j]) + turn_list(pre_slist[j + 1])
            elif j == pre_slen-1:
                coal = turn_list(pre_slist[j-1]) + turn_list(pre_slist[j])
                j = j-1
            else:
                right_coal = turn_list(pre_slist[j]) + turn_list(pre_slist[j+1])
                left_coal = turn_list(pre_slist[j-1]) + turn_list(pre_slist[j])
                right_slist = pre_slist[:j]  # elems before j
                right_slist.append(right_coal)
                if j + 2 < pre_slen:
                    right_slist = right_slist + pre_slist[j + 2:]  # elems after j+1
                if j-2 > -1:
                    left_slist = pre_slist[:j-1]  # elems before j
                else:
                    left_slist = []
                left_slist.append(left_coal)
                if j + 1 < pre_slen:
                    left_slist = left_slist + pre_slist[j + 1:]  # elems after j+1
                
                right_scores_now, _ = compute_scores(right_slist, j, feature,
                                                    a_len, model, args, device)
                left_scores_now, _ = compute_scores(left_slist, j-1, feature,
                                                    a_len, model, args, device)
                right_s= right_scores_now[0:1] - np.sum(right_scores_now[1:])
                left_s = left_scores_now[0:1] - np.sum(left_scores_now[1:])
                if right_s > left_s:
                    coal = right_coal
                else:
                    coal = left_coal
                    j = j-1
            now_slist = pre_slist[:j]  # elems before j
            now_slist.append(coal)
            if j + 2 < pre_slen:
                now_slist = now_slist + pre_slist[j + 2:]  # elems after j+1

            pre_slist = now_slist

            scores_now, logits = compute_scores(pre_slist, j, feature,           # logits: the output of the sentence
                                                    a_len, model, args, device)
            if len(coal) != a_len:
                interaction_contribution = scores_now[0:1] - np.sum(scores_now[1:])
                important_sen.append((interaction_contribution,coal))

        """ sort contribution """
        important_sen = sorted(important_sen, key=lambda item: item[0], reverse=True)
        phrase_list = []
        pre_items = []
        score_list = []
        for score, items in important_sen:
            if not set(items) == set(pre_items):
                phrase_list.append(items)
                score_list.append(score)
                pre_items = items
        for feaidx in phrase_list:
            # single word
            if len(feaidx) == 1:
                if wordvocab[feature[feaidx[0]]] != '<pad>' and wordvocab[feature[feaidx[0]]] != '<unk>':
                    fileobject.write(str(feaidx[0]))
                    fileobject.write(' ')
            else:
                # span
                fea_end = -1
                for fea in feaidx[-1::-1]:
                    if wordvocab[feature[fea]] != '<pad>' and wordvocab[feature[fea]] != '<unk>':
                        fea_end = fea
                        break
                if fea_end > -1 and fea_end>feaidx[0]:
                    fileobject.write(str(feaidx[0]))
                    fileobject.write('-')
                    fileobject.write(str(fea_end))
                    fileobject.write(' ')
        fileobject.write(' >> ')

        scores_aft, _ = compute_scores([list(range(a_len))], 0, feature,
                                                    a_len, model, args, device)
        hier_tree[a_len-1] = []
        hier_tree[a_len-1].append((list(range(a_len)),float(scores_aft[0:1])))

        # prediction label
        if np.argmax(logits, axis=1)[0] == 0:
            fileobject.write('0')
            pred_label = 0

        else:
            fileobject.write('1')
            pred_label = 1

        fileobject.write('\n')
        fileobject.flush()

        acc += (np.argmax(logits, axis=1)[0] == batch.label.cpu().numpy()[0]).sum()

        print(count, " texts have been explained, and the current length of this text is ", len(batch.sentence))
        if args.visualize != -2 :
            visualize_explanation(batch, a_len, hier_tree, pred_label, wordvocab, fontsize=8, tag=count)
            print("Visualize number {:.0f} text".format(args.visualize))
    if count == 0:
        print("The length of the input text is not appropriate, please input again!")
        return 0, 0
    else:
        return acc/count, count

def visualize_explanation(batch, a_len, hier_tree, pred_label, wordvocab, fontsize=8, tag=''):
    max_level = a_len -1
    levels = max_level
    vals = np.array([fea[1] for level in range(levels) for fea in hier_tree[level]])
    min_val = np.min(vals)
    max_val = np.max(vals)

    # max_color = max_val if max_val>0 else -max_val+0.1
    # min_color = min_val if min_val<0 else -min_val-0.1
    max_color = 1
    min_color = -1
    cnorm = mpl.colors.Normalize(vmin=min_color, vmax=max_color, clip=False)

    if pred_label == 1: #1 stands for positive
        cmapper = mpl.cm.ScalarMappable(norm=cnorm, cmap='coolwarm')
    else: #0 stands for negative
        # cmap='RdYlBu_r'
        cmapper = mpl.cm.ScalarMappable(norm=cnorm, cmap='coolwarm_r')                

    words = batch.sentence.cpu().numpy()
    nwords = words.shape[0]
    fig, ax = plt.subplots(figsize=(12, 7))
    # fig, ax = plt.subplots(figsize=(25, 20))
    ax.xaxis.set_visible(False)

    ylabels = ['Step '+str(idx) for idx in range(max_level+1)]
    ax.set_yticks(list(range(0, max_level+1)))                                 # axis
    ax.set_yticklabels(ylabels,fontsize=12)                                         # label
    ax.set_ylim(max_level+0.5, 0-0.5)                                          # range

    sep_len = 0.3
    for key in range(levels+1):
        for fea in hier_tree[key]:
            # fea[0]:list exa:[2, 3] type(fea[0])=list
            len_fea = 1 if type(fea[0]) == int else len(fea[0])     #if=true,len_fea = 1; else len_fea = len(fea[0])
            start_fea = fea[0] if type(fea[0])==int else fea[0][0]
            start = sep_len * start_fea + start_fea + 0.5
            width = len_fea + sep_len * (len_fea - 1)
            fea_color = cmapper.to_rgba(fea[1])                     # importance score to RGBA color
            r, g, b, _ = fea_color
            
            c = ax.barh(key, width=width, height=0.5, left=start, color=fea_color)      # Horizontal bar chart
            
            text_color = 'white' if r * g * b < 0.2 else 'black'                        # font color
            #         text_color = 'black'
            word_idxs = fea[0]
            for i, idx in enumerate(word_idxs):
                word_pos = start + sep_len * (i) + i + 0.5                              # word position
                word_str = wordvocab[batch.sentence[idx]]                                   # index to word
                ax.text(word_pos, key, word_str, ha='center', va='center',              # comment
                        color=text_color, fontsize=13)
                word_pos += sep_len
            start += (width + sep_len)
    cb = fig.colorbar(cmapper, ax=ax)                                                   # colorbar                                 
    cb.ax.tick_params(labelsize=12)
    plt.savefig('visualization_sentence_{}.png'.format(tag))

if __name__ == "__main__":
    if args.visualize > -1:
        start_pos = args.visualize
        end_pos = start_pos + 1
    else:
        start_pos = args.start_pos
        end_pos = args.end_pos

    """ Path for well trained model；Path for the explanation """
    trainedmodel_dir = "./TrainedModel/cnn/" + args.task_name + "/model.pt"
    explain_path = "./Explain_results/cnn/" + args.task_name + "_result.txt"

    """ dataloader """
    data = DATA(args, tokenizer)
    data_iter = data.data_iter

    """ load vocab """
    wordvocab = data.TEXT.vocab.itos

    """ load well trained model """
    if args.gpu > -1:
        with open(trainedmodel_dir, 'rb') as f:
            model = torch.load(f)
        model.to(device)         
    else:
        with open(trainedmodel_dir, 'rb') as f:
            model = torch.load(f, map_location='cpu')

    print("*"*10, "Both of data and model are loaded well, let`s explain", "*"*10)
    f = open(explain_path, 'w')
    acc, count = explain(args, model, data_iter, start_pos, end_pos, f, args.visualize)
    f.close()
    print("\nExplanation is over, and {:.0f} texts are explained, and the average accuracy of these texts is {:.6f} ".format(count, acc))
import torch
import pickle
import json
import numpy as np
from rouge import Rouge
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = SentenceTransformer('answerdotai/ModernBERT-base').to(device)
vlsp_dataset_path = './MultiNews_Dataset'

with open('./MultiNews_Dataset/sent_embeddings.pkl', 'rb') as f:
    vec_list = pickle.load(f)


def getRouge2(ref, pred, kind):  # tokenized input
    try:
        return round(Rouge().get_scores(pred.lower(), ref.lower())[0]['rouge-2'][kind], 4)
    except ValueError:
        return 0.0

class Cluster:
    def __init__(self, cluster, sent_texts, sent_vecs, spanIds, doc_lens, doc_sec_mask, sec_sen_mask, golden, threds=0.5):
        assert len(sent_vecs) == len(sent_texts)
        self.docnum = len(doc_sec_mask)
        self.secnum = len(sec_sen_mask)
        self.feature = torch.cat((torch.stack(sent_vecs, dim=0), torch.zeros((self.secnum+self.docnum+1, sent_vecs[0].shape[0]))), dim=0)
        self.adj = torch.from_numpy(mask_to_adj(doc_sec_mask, sec_sen_mask)).float()
        self.sent_text = sent_texts
        self.spanIds = spanIds
        self.doc_lens = doc_lens
        self.cluster = cluster
        self.golden = golden
        # self.goldenVec = get_cluster_vec(golden)
        self.score = torch.Tensor([getRouge2(golden, s, 'p') for s in sent_texts]).float()
        self.score_onehot = (self.score >= threds).float()
        self.init_node_vec()
        self.feature = self.feature.float()

    def init_node_vec(self):
        docnum, secnum = self.docnum, self.secnum
        for i in range(-secnum-docnum-1, -docnum-1):
            mask = self.adj[i].clone()
            mask[-secnum-docnum-1:] = 0
            self.feature[i] = torch.mean(self.feature[mask.bool()], dim=0)
        for i in range(-docnum-1, -1):
            mask = self.adj[i].clone()
            mask[-docnum-1:] = 0
            self.feature[i] = torch.mean(self.feature[mask.bool()], dim=0)
        self.feature[-1] = torch.mean(self.feature[-docnum-1:-1], dim=0)

def mask_to_adj(doc_sec_mask, sec_sen_mask):
    sen_num = sec_sen_mask.shape[1]
    sec_num = sec_sen_mask.shape[0]
    doc_num = doc_sec_mask.shape[0]
    adj = np.zeros((sen_num+sec_num+doc_num+1, sen_num+sec_num+doc_num+1))
    # section connection
    adj[-sec_num-doc_num-1:-doc_num-1, 0:-sec_num-doc_num-1] = sec_sen_mask
    adj[0:-sec_num-doc_num-1, -sec_num-doc_num-1:-doc_num-1] = sec_sen_mask.T
    for i in range(0, doc_num):
        doc_mask = doc_sec_mask[i]
        doc_mask = doc_mask.reshape((1, len(doc_mask)))
        adj[sen_num:-doc_num-1, sen_num:-doc_num-1] += doc_mask * doc_mask.T
    # doc connection
    adj[-doc_num-1:-1, -sec_num-doc_num-1:-doc_num-1] = doc_sec_mask
    adj[-sec_num-doc_num-1:-doc_num-1, -doc_num-1:-1] = doc_sec_mask.T
    adj[-doc_num-1:, -doc_num-1:] = 1

    #build sentence connection
    for i in range(0, sec_num):
        sec_mask = sec_sen_mask[i]
        sec_mask = sec_mask.reshape((1, len(sec_mask)))
        adj[:sen_num, :sen_num] += sec_mask * sec_mask.T
    return adj

class GraphDataset(Dataset):
    def __init__(self, mode='200_train', limit=200):
        self.clusters, self.goldens = [], []
        self.prepare_data(mode, limit)

    def prepare_data(self, mode, limit):
        with open(f'{vlsp_dataset_path}/{mode}_processed.json', 'r', encoding='utf-8') as fp:
            clusTrees = json.load(fp)
        with open(f'{vlsp_dataset_path}/{mode}_secnum.pkl', 'rb') as fp:
            seclist = pickle.load(fp)

        for cluster in tqdm(range(0, limit)):
            sents, sentVecs, secIDs, spanIds, doc_lens = [], [], [], [], []
            clusTree = clusTrees[cluster]

            secnum = max(seclist[cluster][len(seclist[cluster]) - 1].values()) + 1
            sentnum = sum([len(doc.values()) for doc in seclist[cluster].values()])
            doc_sec_mask = np.zeros((len(clusTree['docs']), secnum))
            sec_sen_mask = np.zeros((secnum, sentnum))
            cursec, cursent = 0, 0
            for d, doc in enumerate(clusTree['docs']):
                sent_list = []
                for p in doc['paragraphs']:
                    sent_list += p
                doc_lens.append(len(sent_list))
                doc_endsec = max(seclist[cluster][d].values())
                doc_sec_mask[d][cursec:doc_endsec + 1] = 1
                cursec = doc_endsec + 1
                for s, sent in enumerate(sent_list):
                    sents.append(sent)
                    sentVecs.append(torch.from_numpy(vec_list[sent]))
                    spanIds.append((d, s))
                    sec_sen_mask[seclist[cluster][d][s], cursent] = 1
                    cursent += 1

            self.clusters.append(Cluster(cluster, sents, sentVecs, spanIds, doc_lens, doc_sec_mask, sec_sen_mask, ' '.join(clusTree['summary'])))

            sentVecs = [torch.from_numpy(vec_list[sent]) for sent in clusTree['summary']]
            self.goldens.append(Cluster(cluster, clusTree['summary'], sentVecs, [], [len(clusTree['summary'])], np.ones((1, 1)), np.ones((1, len(clusTree['summary']))), 'abc'))

    def __getitem__(self, index):
        data, golden_graph = self.clusters[index], self.goldens[index]
        return {
            'feature': data.feature,
            'labels': data.score,
            'adj': data.adj,
            'doc_lens': data.doc_lens,
            'docnum': data.docnum,
            'secnum': data.secnum,
            'golden': data.golden,
            'sent_text': data.sent_text,
            'spanIds': data.spanIds,
            'golden_feature': golden_graph.feature,
            'golden_adj': golden_graph.adj,
            'golden_doc_lens': golden_graph.doc_lens,
            'cluster': data.cluster
        }

    def __len__(self):
        return len(self.clusters)

def getPositionEncoding(pos, d=768, n=10000):
    P = np.zeros(d)
    for i in np.arange(int(d / 2)):
        denominator = np.power(n, 2 * i / d)
        P[2 * i] = np.sin(pos / denominator)
        P[2 * i + 1] = np.cos(pos / denominator)
    return P

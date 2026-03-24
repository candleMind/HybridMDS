import torch
import pickle
import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk

nltk.download('punkt_tab')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = SentenceTransformer('answerdotai/ModernBERT-base').to(device)


def clean_text(text):
    if re.search(r'\(Click Prev or Next.*?images\.\)', text):
        return ""
    if re.search(r'\(Photo: .*?\)', text):
        return ""
    if re.search(r'http?[s]?:\/\/?[-a-zA-Z0-9@:%._\+~#=]{1,256}[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)',
                 text):
        return ""
    if re.search(r'[^@ \t\r\n]+@[^@ \t\r\n]+\.[^@ \t\r\n]+', text):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def divideSection(doc, tf, lda_model):
    sent_para, para_sec, sent_sec = {}, {}, {}
    paras = []
    sentcnt = 0
    for i, para in enumerate(doc['paragraphs']):
        paras.append(' '.join(para).lower())
        for ii, sent in enumerate(para):
            sent_para[sentcnt + ii] = i
        sentcnt += len(para)

    # prepare para_sec
    # tf, lda_model = cate_models[category]
    X = tf.transform(paras)
    lda_top = lda_model.transform(X)
    for i, para_top in enumerate(lda_top):
        para_sec[i] = para_top.argmax()

    # output sent_sec
    for k, v in sent_para.items():
        sent_sec[k] = para_sec[v]
    return sent_sec


def divideSection_limitlen(doc, limitlen=250):
    sent_para, para_sec, sent_sec = {}, {}, {}
    paras = []
    sentcnt = 0
    for i, para in enumerate(doc['paragraphs']):
        paras.append(' '.join(para).lower())
        for ii, sent in enumerate(para):
            sent_para[sentcnt + ii] = i
        sentcnt += len(para)

    # prepare para_sec
    cur_len, stt = 0, 0
    for i, p in enumerate(paras):
        cur_len += len(p.split())
        if cur_len > limitlen:
            cur_len = len(p.split())
            stt += 1
        para_sec[i] = stt

    # output sent_sec
    for k, v in sent_para.items():
        sent_sec[k] = para_sec[v]
    return sent_sec

def train_LDA_model():
    paras = []
    with open('MultiNews_Dataset/200_train_processed.json', 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    for clus in clusters:
        for doc in clus['docs']:
            for para in doc['paragraphs']:
                paras.append(' '.join(para).lower())

    tf = TfidfVectorizer(min_df=2, max_df=1.0, max_features=3000, sublinear_tf=True, stop_words='english')
    X = tf.fit_transform(paras)
    lda_model = LatentDirichletAllocation(n_components=4, learning_method='online', random_state=42, max_iter=1)
    lda_model.fit(X)
    with open('MultiNews_Dataset/LDA_models_4top.pkl', mode='wb') as fp:
        pickle.dump((tf, lda_model), fp)
    return tf, lda_model


def main():
    for dataset in ['train', 'val', 'test']:
        if dataset == 'train':
            k = 200
        elif dataset == 'val':
            k = 100
        else:
            k = 150

        with open(f"MultiNews_Dataset/{dataset}.src.cleaned", "r", encoding="utf-8") as f:
            data = f.readlines()
        with open(f"MultiNews_Dataset/{dataset}.tgt", "r", encoding="utf-8") as f:
            summs = f.readlines()

        clusters = []
        for d, summ in zip(data, summs):
            d = d.strip().replace("NEWLINE_CHAR", "\n")
            docs = d.split("|||||")
            tokens = [len(doc.split(" ")) for doc in docs]
            clusters.append({'docs': docs, 'tokens': tokens, 'summary': summ[1:].strip()})

        # Select k clusters for subset
        filter_clusters = []
        for cluster in clusters:
            if min(cluster['tokens']) >= 400 and max(cluster['tokens']) < 1500 and 2 <= len(cluster['docs']) <= 6:
                filter_clusters.append(cluster)
        filter_clusters = random.sample(filter_clusters, k)
        with open(f'MultiNews_Dataset/{k}_{dataset}.json', 'w', encoding='utf-8') as f:
            json.dump(filter_clusters, f)

        # Clean and preprocess data
        clusters = filter_clusters
        processed_clusters = []
        for i, clus in enumerate(clusters):
            new_clus = {'cluster_id': i, 'docs': []}
            for doc in clus['docs']:
                paras = [p.strip() for p in doc.split('\n') if p.strip()]
                new_clus['docs'].append({'paragraphs': []})

                for para in paras:
                    sents = nltk.sent_tokenize(para)
                    cleaned_sents = [clean_text(s.strip()) for s in sents if s.strip()]
                    cleaned_sents = [s for s in cleaned_sents if len(s.split()) >= 4]
                    if len(cleaned_sents) == 0:
                        continue
                    new_clus['docs'][-1]['paragraphs'].append(cleaned_sents)

            summary = re.sub(r'\s+', ' ', clus['summary']).strip()
            sents = nltk.sent_tokenize(summary)
            new_clus['summary'] = [s.strip() for s in sents if s.strip()]
            processed_clusters.append(new_clus)
        with open(f'MultiNews_Dataset/{k}_{dataset}_processed.json', 'w', encoding='utf-8') as f:
            json.dump(processed_clusters, f)

    # Prepare LDA section division
    tf, lda_model = train_LDA_model()
    for dataset in ['train', 'val', 'test']:
        if dataset == 'train':
            k = 200
        elif dataset == 'val':
            k = 100
        else:
            k = 150
        seclist = {}
        with open(f"MultiNews_Dataset/{k}_{dataset}_processed.json", "r", encoding="utf-8") as f:
            clusters = json.load(f)
        for clusid, clus in enumerate(clusters):
            seclist[clusid] = {}
            for d, doc in enumerate(clus['docs']):
                seclist[clusid][d] = divideSection(doc, tf, lda_model)

            secnum = 0
            for k, val_dict in seclist[clusid].items():
                vals = set(val_dict.values())
                for ki, vi in val_dict.items():
                    for i, v in enumerate(vals):
                        if vi == v:
                            val_dict[ki] = i + secnum
                            break
                seclist[clusid][k] = val_dict
                secnum += len(vals)
        with open(f"MultiNews_Dataset/{k}_{dataset}_secnum.pkl", 'wb') as f:
            pickle.dump(seclist, f)

    # Get sentence embeddings
    clusters = []
    with open('MultiNews_Dataset/200_train_processed.json', 'r', encoding='utf-8') as f:
        clusters += json.load(f)
    with open('MultiNews_Dataset/100_val_processed.json', 'r', encoding='utf-8') as f:
        clusters += json.load(f)
    with open('MultiNews_Dataset/150_test_processed.json', 'r', encoding='utf-8') as f:
        clusters += json.load(f)
    sent_list = []
    for clus in clusters:
        for doc in clus['docs']:
            for para in doc['paragraphs']:
                sent_list.extend(para)
        sent_list.extend(clus['summary'])
    unique_sents = set(sent_list)
    print(len(unique_sents))
    with open('MultiNews_Dataset/unique_sents.json', 'w', encoding='utf-8') as f:
        json.dump(list(unique_sents), f)

    with open('./MultiNews_Dataset/unique_sents.json', 'r', encoding='utf-8') as f:
        sents = json.load(f)
    sentVecs = bert.encode(sents, batch_size=128, show_progress_bar=True)
    sentVecs_dict = {sent: sentVecs[i] for i, sent in enumerate(sents)}
    with open('./MultiNews_Dataset/sent_embeddings.pkl', 'wb') as f:
        pickle.dump(sentVecs_dict, f)

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union
from scipy.stats import entropy
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
COLORS = ('r','g','b','c','m','y','k')

def mean_cls_entropy(attn, layer=-1):
    attn = attn[layer]
    attn = torch.mean(attn, dim=0)[0].numpy()
    terms = np.array([-a*np.log(a, out=np.zeros_like(a), where=(a!=0)) for a in attn])
    #terms = np.nan_to_num(terms)
    return sum(terms), terms

def jensen_shannon(
        t1 : Union[torch.Tensor, np.ndarray],
        t2 : Union[torch.Tensor, np.ndarray]
    ):
    print(t1)
    print(t2)
    
    m = (t1 + t2) * 0.5
    print(m)
    div1 = 0.5 * F.kl_div(t1, m, reduction='batchmean')
    div2 = 0.5 * F.kl_div(t2, m, reduction='batchmean')
    return float(div1 + div2)

def KLD(
    t1 : Union[torch.Tensor, np.ndarray],
    t2 : Union[torch.Tensor, np.ndarray]
):
    return F.kl_div(t1, t2, reduction='batchmean')

def tvd(
        t1 : Union[torch.Tensor, np.ndarray],
        t2 : Union[torch.Tensor, np.ndarray]
    ):

    diff = torch.linalg.norm(t1-t2)
    return 0.5*sum(diff)

def plot_tsne(eid, psgs, dim=2, identifier : list = None):
    all_embs = [emb[1].numpy() if isinstance(emb[1], torch.Tensor) else emb[1] for emb in psgs]
    #sub_embs = all_embs[:2] + random.sample(all_embs[2:], k=248)
    all_embs = np.stack(all_embs, axis=0)
    #sub_embs = np.stack(sub_embs, axis=0)
    print(all_embs.shape)

    emb_types = set(identifier)

    tsne_plot = TSNE( 
        n_components=dim,
        perplexity=50.0,
        early_exaggeration=24.0,
        n_iter=500,
        n_iter_without_progress=150,
        init='pca',
        n_jobs=6,
    )
    mapped_embs = tsne_plot.fit_transform(all_embs)
    print(mapped_embs.shape)
    print(mapped_embs[0].shape)
    colors = identifier[:all_embs.shape[1]]
    marker_sizes = identifier[:all_embs.shape[1]]
    hue_order=[0,1,2]
    zorder = [50,100,150]

    for k, etype in enumerate(emb_types):
        is_etype = [idf == etype for i, idf in enumerate(identifier)]
        embs = [mapped_embs[i] for i, flag in enumerate(is_etype) if flag]
        embs = np.stack(embs, axis=0).T
        vis = [k for i in is_etype if i]
        sns.scatterplot(
            x=embs[0],
            y=embs[1],
            hue=vis,
            #hue_order=hue_order,
            size=vis,
            marker=',',
            palette={0:'#DB9586',1:'#000000',2:'#450CEA', 3:'#258503'},
            sizes={0:3,1:15,2:15,3:15}
        )

    #plt.scatter(mapped_embs[0], mapped_embs[1], c=colors, marker=',')
    #sns.scatterplot(
    #    x=mapped_embs[0],
    #    y=mapped_embs[1],
    #    hue=colors,
    #    hue_order=hue_order,
    #    size=marker_sizes,
    #    marker=',',
    #    palette={0:'#DB9586',1:'#450CEA',2:'#258503'},
    #    sizes={0:3,1:15,2:15}
    #    #palette='husl',
    #)
    plt.savefig(f'tsne_{eid}.png')
    plt.close()

def plot_tsne_aggregate(psgs, dim=2):
    all_embs = list()
    identifier = list()
    for i, emb in enumerate(psgs):
        all_embs.extend([em[1].numpy() if isinstance(em[1], torch.Tensor) else em[1] for em in emb])
        identifier.extend([i] * len(emb))
    all_embs = np.stack(all_embs, axis=0)
    
    tsne_plot = TSNE( 
        n_components=dim,
        perplexity=50.0,
        early_exaggeration=24.0,
        n_iter=500,
        n_iter_without_progress=150,
        init='pca',
        n_jobs=6,
    )
    mapped_embs = tsne_plot.fit_transform(all_embs).T
    
    colors = identifier
    #marker_sizes = identifier
    palet = {i : random.random() for i in range(len(psgs))}
    sns.scatterplot(
        x=mapped_embs[0],
        y=mapped_embs[1],
        hue=colors,
        hue_order=colors,
        marker=',',
        palette=palet,
        #size=marker_sizes,
        #sizes={0:3,1:15}
        #palette='husl',
    )
    plt.savefig(f'tsne.png')
    plt.close()
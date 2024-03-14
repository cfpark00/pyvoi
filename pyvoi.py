import numpy as np
import torch
import numpy.typing as npt


def VI(labels1: npt.NDArray[np.int32],labels2: npt.NDArray[np.int32],torch: bool=True,device: str="cpu",return_split_merge: bool=False):
    """
    Calculates the Variation of Information between two clusterings.

    Arguments:
    labels1: flat int32 array of labels for the first clustering
    labels2: flat int32 array of labels for the second clustering
    torch: whether to use torch, default:True
    device: device to use for torch, default:"cpu"
    return_split_merge: whether to return split and merge terms, default:False

    Returns:
    vi: variation of information
    vi_split: split term of variation of information
    vi_merge: merge term of variation of information
    splitters(optional): labels of labels2 which are split by labels1. splitters[i,0] is the contribution of the i-th splitter to the VI and splitters[i,1] is the corresponding label of the splitter
    mergers(optional): labels of labels1 which are merging labels from labels2. mergers[i,0] is the contribution of the i-th merger to the VI and mergers[i,1] is the corresponding label of the merger
    """
    if torch:
        return VI_torch(labels1,labels2,device=device,return_split_merge=return_split_merge)
    else:
        return VI_np(labels1,labels2,return_split_merge=return_split_merge)

def VI_np(labels1,labels2,return_split_merge=False):
    assert len(labels2)==len(labels1)
    size=len(labels2)

    mutual_labels=(labels1.astype(np.uint64)<<32)+labels2.astype(np.uint64)

    sm_unique,sm_inverse,sm_counts=np.unique(labels2,return_inverse=True,return_counts=True)
    fm_unique,fm_inverse,fm_counts=np.unique(labels1,return_inverse=True,return_counts=True)
    _,mutual_inverse,mutual_counts=np.unique(mutual_labels,return_inverse=True,return_counts=True)

    terms_mutual = -np.log(mutual_counts/size)*mutual_counts/size
    terms_mutual_per_count=terms_mutual[mutual_inverse]/mutual_counts[mutual_inverse]
    terms_sm = -np.log(sm_counts/size)*sm_counts/size
    terms_fm = -np.log(fm_counts/size)*fm_counts/size
    if not return_split_merge:
        terms_mutual_sum=np.sum(terms_mutual_per_count)
        vi_split=terms_mutual_sum-terms_sm.sum()
        vi_merge=terms_mutual_sum-terms_fm.sum()
        vi=vi_split+vi_merge
        return vi,vi_split,vi_merge

    vi_split_each=np.zeros(len(sm_unique))
    np.add.at(vi_split_each,sm_inverse,terms_mutual_per_count)
    vi_split_each-=terms_sm
    vi_merge_each=np.zeros(len(fm_unique))
    np.add.at(vi_merge_each,fm_inverse,terms_mutual_per_count)
    vi_merge_each-=terms_fm

    vi_split=np.sum(vi_split_each)
    vi_merge=np.sum(vi_merge_each)
    vi=vi_split+vi_merge

    i_splitters=np.argsort(vi_split_each)[::-1]
    i_mergers=np.argsort(vi_merge_each)[::-1]

    vi_split_sorted=vi_split_each[i_splitters]
    vi_merge_sorted=vi_merge_each[i_mergers]

    splitters=np.stack([vi_split_sorted,sm_unique[i_splitters]],axis=1)
    mergers=np.stack([vi_merge_sorted,fm_unique[i_mergers]],axis=1)
    return vi,vi_split,vi_merge,splitters,mergers

def VI_torch(labels1,labels2,device="cpu",return_split_merge=False):
    assert len(labels2)==len(labels1)
    size=len(labels2)
    labels1=torch.tensor(labels1,dtype=torch.int64,device=device)
    labels2=torch.tensor(labels2,dtype=torch.int64,device=device)

    mutual_labels=(labels1<<32)+labels2

    sm_unique,sm_inverse,sm_counts=torch.unique(labels2,sorted=False,return_inverse=True,return_counts=True)
    fm_unique,fm_inverse,fm_counts=torch.unique(labels1,sorted=False,return_inverse=True,return_counts=True)
    _,mutual_inverse,mutual_counts=torch.unique(mutual_labels,sorted=False,return_inverse=True,return_counts=True)

    terms_mutual = -torch.log(mutual_counts/size)*mutual_counts/size
    terms_mutual_per_count=terms_mutual[mutual_inverse]/mutual_counts[mutual_inverse]
    terms_sm = -torch.log(sm_counts/size)*sm_counts/size
    terms_fm = -torch.log(fm_counts/size)*fm_counts/size
    if not return_split_merge:
        terms_mutual_sum=torch.sum(terms_mutual_per_count)
        vi_split=terms_mutual_sum-terms_sm.sum()
        vi_merge=terms_mutual_sum-terms_fm.sum()
        vi=vi_split+vi_merge
        return vi,vi_split,vi_merge
    
    vi_split_each=torch.zeros(len(sm_unique),dtype=torch.float32,device=device)
    vi_split_each.index_put_((sm_inverse,),terms_mutual_per_count, accumulate=True)
    vi_split_each-=terms_sm
    vi_merge_each=torch.zeros(len(fm_unique),dtype=torch.float32,device=device)
    vi_merge_each.index_put_((fm_inverse,),terms_mutual_per_count, accumulate=True)
    vi_merge_each-=terms_fm

    vi_split=torch.sum(vi_split_each)
    vi_merge=torch.sum(vi_merge_each)
    vi=vi_split+vi_merge

    i_splitters=torch.argsort(vi_split_each,descending=True)
    i_mergers=torch.argsort(vi_merge_each,descending=True)

    vi_split_sorted=vi_split_each[i_splitters]
    vi_merge_sorted=vi_merge_each[i_mergers]

    splitters=torch.stack([vi_split_sorted,sm_unique[i_splitters]],dim=1)
    mergers=torch.stack([vi_merge_sorted,fm_unique[i_mergers]],dim=1)
    return vi,vi_split,vi_merge,splitters,mergers
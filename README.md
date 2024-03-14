# pyvoi: Variation of Information in numpy/torch

Calculate the Variation of Information for two clusterings

References:
Meila, Marina (2003). Comparing Clusterings by the Variation of Information. Learning Theory and Kernel Machines: 173–187

## Installation

    pip install python-voi

## Usage

By default, pyvoi uses torch to compute the variation of information.

### To get VI values only
    import pyvoi
    labels1=[0,1,1,2,4]
    labels2=[0,2,3,4,4]
    vi,vi_split,vi_merge=pyvoi.VI(labels1,labels2)
    print(vi,vi_split,vi_merge)
    # total vi, split vi, merge vi
    # tensor(0.5545) tensor(0.2773) tensor(0.2773)
### To get VI components
    import pyvoi
    labels1=[0,1,1,2,4]
    labels2=[0,2,3,4,4]
    vi,vi_split,vi_merge,splitters,mergers=pyvoi.VI(labels1,labels2,return_split_merge=True)
    print(vi,vi_split,vi_merge)
    # (same as above) total vi, split vi, merge vi
    # tensor(0.5545) tensor(0.2773) tensor(0.2773)
    
    print(splitters)
    # vi contribution, label in label2
    #tensor([[0.2773, 4.0000],
    #    [0.0000, 3.0000],
    #    [0.0000, 2.0000],
    #    [0.0000, 0.0000]])
    
    print(mergers)
    # vi contribution, label in label1
    #tensor([[0.2773, 1.0000],
    #    [0.0000, 4.0000],
    #    [0.0000, 2.0000],
    #    [0.0000, 0.0000]])

### Using cuda

    import pyvoi
    labels1=[0,1,1,2,4]
    labels2=[0,2,3,4,4]
    vi,vi_split,vi_merge=pyvoi.VI(labels1,labels2,device="cuda")
    print(vi,vi_split,vi_merge)
    #tensor(0.5545, device='cuda:0') tensor(0.2773, device='cuda:0') tensor(0.2773, device='cuda:0')

### Using numpy

    import numpy as np
    import pyvoi
    labels1=np.array([0,1,1,2,4])
    labels2=np.array([0,2,3,4,4])
    vi,vi_split,vi_merge=pyvoi.VI(labels1,labels2,torch=False)
    print(vi,vi_split,vi_merge)
    #0.5545177444479561 0.27725887222397794 0.27725887222397816
    
## Contact and Bug Reports
Core Francisco Park: cfpark00@gmail.com

import matplotlib.pyplot as plt
import numpy as np
import torch
def draw(x,title=None,y_lim=None,i=None):
    plt.figure()
    if title:
        plt.title(title)
    if y_lim:
        plt.ylim(y_lim)
    if i ==1:
        plt.plot(range(x.shape[0]),x)
    else:
        plt.bar(range(x.shape[0]),x)
    plt.xticks([])
    plt.yticks([])
    if i is not None:
        import os
        path = '/home/usr/sc2Flow_lab/figures/'
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(f'{path}', f'{i}.png')
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"figure saved to: {full_path}")
        plt.show()

def list2vector(ids,values, show=True, title=None, y_lim=None,i=None):
    values =  (values + 1) / 2 * 6
    vector = np.zeros(2001)
    vector[ids] = values.cpu().numpy()
    vector = vector[1:]
    draw(values[:sum(ids!=0)],title,y_lim,i)
    return vector

def list2tensor(ids,values):
    values =  (values + 1) / 2 * 6
    vector = torch.zeros(2001)
    vector[ids] = values.to(torch.float32)
    vector = vector[1:]
    return vector

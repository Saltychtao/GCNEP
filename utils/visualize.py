import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def svd(M):
    u,s,vt = np.linalg.svd(M,full_matrices=False)
    V = vt.T
    S = np.diag(s)
    return np.dot(u[:,:2],np.dot(S[:2,:2],V[:,:2].T))


def plot(embedding,fname,labels):
    embedding_points = svd(embedding)
    x = embedding_points[:,0]
    y = embedding_points[:,1]
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['z'] = labels
    sns.scatterplot(x='x',y='y',data=df,hue='z',s=10)
    plt.savefig(fname)
    plt.clf()


if __name__ == '__main__':
    embedding = np.random.rand(5000,10)
    fname = 'plt.png'
    labels = np.random.randint(0,2,size=(5000))
    plot(embedding,fname,labels) 

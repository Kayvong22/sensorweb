import scipy.optimize as sci
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()
import numpy as np
from matplotlib.patches import Ellipse
import pandas as pd

def Standardize(x):
    mean = np.mean(x)
    standard_deviation = np.std(x)

    return (x - mean) / standard_deviation


class cluster_gaussianMM(object):
    """docstring for gaussianMM."""

    def __init__(self, Kcoef):
        self.Kcoef = Kcoef
        self.X = None
        self.Z = None
        self.labels = None
        self.centroids = None
        self.lenState = None
        self.slenState = None
        self.applP = None
        self.sapplP = None
        self.gmm = None

    def sample(self):
        """Prepares the input sample from the OMP output (K strongest atoms)."""

        self.lenState = np.count_nonzero(self.Kcoef, axis=0)
        # self.lenState = 1 + 3 * np.log(self.lenState)
        self.slenState = Standardize(self.lenState)

        self.applP = np.max(self.Kcoef, axis=0)
        self.sapplP = Standardize(self.applP)

        self.X = np.vstack((self.slenState, self.sapplP*3.)).T
        # self.Z = linkage(self.X, 'ward')

    def retrieve_clus(self, number_component,random_state=None,cov='full',seed=None):
        """Retrieve the number of cluster knowing the number of clusters."""

        self.sample()

        self.gmm = GaussianMixture(n_components=number_component,covariance_type=cov,random_state=seed,warm_start=True).fit(self.X)
        self.REFIND = self.gmm.predict(self.X)

    def plotclust(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.REFIND, s=40, cmap='viridis', zorder=2)

    def calc_centroids(self):
        
        self.centroids=pd.DataFrame(np.array([np.mean(np.array([np.concatenate([[0],np.pad(np.trim_zeros(j),(0,int(999-len(np.trim_zeros(j)))),'constant')]) 
            for j in np.transpose(self.Kcoef[:,np.where(self.REFIND==i)[0]])]),axis=0) for i in np.unique(self.REFIND)]),index=np.unique(self.REFIND))


def plot_meanCF(data,clusters,ylim=[0.0,90],centroids=None,order=None,ylab='',xlab='',title=None):
    '''
    plot the mean per cluster or the centroid and the percentiles

    data: the dataset N*M as a numpy array
    centroids: centroids as a 2D numpy array N*M*K
    clusters: list of index of assignment (assigned)
    date: the date to ve written on the file
    ylim: limits of the y axis
    name: the name of the output file   
    '''                                                
    
    maxClusters=max(clusters)+1
    
    #days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    nonZeroMax=np.max(np.count_nonzero(data, axis=1))
    x=np.arange(0,nonZeroMax+2,1)
    if divmod(maxClusters,5)[1]!=0:              
        line=int(math.ceil(max(clusters)/5)+1) 
    else:
        line=int(math.ceil(int((maxClusters)/5)))
    #print('line'+str(line))
    f, axs = plt.subplots(line,5,figsize=(5*5,5*line), sharex='col', sharey='row')
    f.subplots_adjust(hspace = .2, wspace=.05)
    axs = axs.ravel()


    for i in range(maxClusters):
        #print(i)
        #axs[i].axis('off')                                    
        if order is not None:
            i2=order[i]
        else:
            i2=i

        try:
            Y = np.array([np.concatenate([[0],np.pad(np.trim_zeros(data[int(j)]),
                (0,int(nonZeroMax+1-len(np.trim_zeros(data[int(j)])))),'constant')]) for j in np.where(clusters==i2)[0]],dtype=float)

            
            if centroids is None:
                centroid=np.nanmedian(Y,axis=0)
            else:
                centroid=centroids[i2]

            if i>=(line-1)*5.:  
               # print(i)
                axs[i].set_xlabel(xlab,size=20)
                for tick in axs[i].xaxis.get_major_ticks():
                    tick.label.set_fontsize(14)   
            if i%5==0: 

                axs[i].set_ylabel(ylab,size=20)
                for tick in axs[i].yaxis.get_major_ticks():
                    tick.label.set_fontsize(14)
            axs[i].set_ylim(ylim) 

                                                             
            axs[i].set_title('Cluster: '+str(i2)+'; Size: '+str(len(Y)),size=15)


            #print(np.shape(Y))

            if len(Y)>1:
                for per in zip([2.5,5,10,25],[97.5,95,90,75]):
                    axs[i].fill_between(x,np.nanpercentile(Y,per[0],axis=0),np.nanpercentile(Y,per[1],axis=0), color="#3F5D7D",alpha=0.3)                                                            
                axs[i].plot(x,centroid,lw=1, color="white")
            else:
                axs[i].plot(x,centroid,lw=1, color="#3F5D7D")
        except ValueError:
            print('skip: '+str(i))
    #day= datetime.strptime(date,'%Y-%m-%d').weekday() 
    #f.suptitle('date = '+date+' ('+days[day]+')',size=15)          
    if title==None: 
        f.savefig('./gmm-K'+ str(maxClusters) +'.png',bbox_inches='tight') 
    else: 
        f.savefig(title +'.png',bbox_inches='tight')
    plt.clf()   
    plt.close()
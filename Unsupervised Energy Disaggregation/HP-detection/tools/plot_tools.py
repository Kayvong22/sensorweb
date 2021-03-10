import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from collections import Counter

# ------------- Plot for the clustering process --------------------------- #

def my_barplot(REFIND, xlabel='', ylabel='', title=''):
    """Bar plot with as input the referenced atoms."""

    df_count_clust = pd.DataFrame(REFIND).from_dict(
        Counter(REFIND), orient='index').sort_values(0, ascending=False).reset_index()
    df_count_clust.columns = ['clust_label', 'amount']

    x = df_count_clust.index.values
    height = df_count_clust.amount.values.flatten()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return plt.bar(x, height, 1)

# ------------- Plot the network community --------------------------------- #

# TODO write the import from resultDic and test it

# # G = cominst.sample_graph
# partition = cominst2.partition
# # p = cominst2.p
#
# pos = community_layout(G, partition)
#
# col = list(cominst.partition.values())
# col = np.asarray(col)
#
# plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
# plt.axis('off')
# nx.draw_networkx_nodes(G, pos, node_size=10,
#                        cmap=plt.cm.RdYlBu, node_color=col)
# nx.draw_networkx_edges(G, pos, alpha=0.1)
# plt.show(G)
# plt.savefig('test.pdf')

# EXPORT the network graph as .pickle
# path2plot = '/Users/pierrewinkler/Documents/nilmplot/experiments_part/exp_AMPd/'
#
# with open(path2plot + 'GraphG' + '.pickle', 'wb') as handle:
#     pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(path2plot + 'posG' + '.pickle', 'wb') as handle:
#     pickle.dump(pos, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open(path2plot + 'colG' + '.pickle', 'wb') as handle:
#     pickle.dump(col, handle, protocol=pickle.HIGHEST_PROTOCOL)

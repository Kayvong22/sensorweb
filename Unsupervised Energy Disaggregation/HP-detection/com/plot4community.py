import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_com_as_signal(community_instance, plot_agg_signal=False):
    """Plot the atoms of each community over time."""
    ComDict = community_instance.ComDict

    llcom = list(ComDict.keys())
    llts = list(ComDict[llcom[0]].keys())

    # Sequence for the colors ...
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

    # # Narrow the list
    llcom = llcom
    llts = llts[0]

    fig1 = plt.figure(figsize=(7, 4), dpi=100)  # preparing the figure

    for i in range(len(llcom)):
        a = llcom[i]

        if type(llts) == str:
            supermat = ComDict[a][llts]  # initialize

        else:
            supermat = ComDict[a][llts[0]]
            for j in range(len(llts)):
                mat = ComDict[a][llts[j]]
                supermat = block_diag(supermat, mat)

        if not supermat.any():
            continue
        else:
            # Plot Plot
            # plt.subplots_adjust
            globals()['ax{}'.format(i + 1)] = fig1.add_subplot(len(llcom), 1, i + 1)
            globals()['ax{}'.format(i + 1)].plot(supermat,
                                                 color=color[i])
            # , label=llcom[i])
            # globals()['ax{}'.format(i + 1)].legend(loc=1, prop={'size': 6})
            # globals()['ax{}'.format(i + 1)].axis([(1440 * 3), (1440 * 6), 0, 0.75])
            globals()['ax{}'.format(i + 1)].set_yticklabels([])
            globals()['ax{}'.format(i + 1)].set_xticklabels([])
            # globals()['ax{}'.format(i + 1)].legend(loc=2, label=llcom[i])

        if plot_agg_signal == True:
            fig2 = plt.figure(figsize=(7, 4), dpi=100)
            axagg = fig2.add_subplot(111)
            axagg.plot(supermat.sum(axis=1))
            axagg.set_yticklabels([])
            axagg.set_xticklabels([])
            plt.show()

        else:
            pass


def plot_network(community_instance, save_name=''):
    """Plot the network from the edges and nodes. Doesn't plot the community."""
    df = community_instance.df

    df['l1'].max()
    df['l1'].min()
    mean = df['l1'].describe()[1]

    sample_graph = community_instance.sample_graph

    elarge = [(u, v) for (u, v, d) in sample_graph.edges(data=True) if d['weight'] > mean]
    esmall = [(u, v) for (u, v, d) in sample_graph.edges(data=True) if d['weight'] <= mean]
    pos = nx.spring_layout(sample_graph)

    # nodes
    nx.draw_networkx_nodes(sample_graph, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(sample_graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(sample_graph, pos, edgelist=esmall, width=6,
                           alpha=0.5, edge_color='b', style='dashed')

    # labels
    nx.draw_networkx_labels(sample_graph, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    # plt.show()  # display
    if save_name == '':
        pass

    else:
        plt.savefig(save_name)  # save as png


def community_layout(g, partition):
    """ Compute the layout for a modular graph.

    Arguments:
        g -- networkx.Graph or networkx.DiGraph instance
            graph to plot
        partition -- dict mapping int node -> int community
            graph partitions

    Returns:
        pos -- dict mapping int node -> (float x, float y) node positions"""

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

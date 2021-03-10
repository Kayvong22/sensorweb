import pickle
from lab.features_extraction import *
from lab.decision_tree import *
from collections import Counter


class labelling_classification():
    """Labelling process via decision tree."""

    def __init__(self, resultDic):
        self.resultDic = resultDic
        self.ll_columns = list(self.resultDic[0].dataframe.drop(
            ['TotalPower_Appliances'], axis=1))

    def classifier_training(self, dictdf_appl_training,
                            path2classifier,
                            train_new_classifier=True):
        """ ."""
        if train_new_classifier == True:
            ll_feat_appl = extract_feature(dictdf_appl_training, samples_per_min)

            self.my_tree = build_tree(ll_feat_appl)

            with open(path2classifier, 'wb') as handle:
                pickle.dump(self.my_tree, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(path2classifier, 'rb') as handle:
                self.my_tree = pickle.load(handle)

    def extract_community_features(self):
        """ ."""
        self.y_com_dict, self.y_truth_dict, self.y_hat_dict = self.build_empty_dict()

        df_y_com = pd.DataFrame(self.y_com_dict)
        self.ll_com = df_y_com.columns
        self.ll_feat_y_com = extract_feature(df_y_com, samples_per_min)

    def labelling_w_classifier(self):
        """ ."""
        ll_best_name = []
        for com in self.ll_com:

            one_com_features = [self.ll_feat_y_com[i] for i in range(
                len(self.ll_feat_y_com)) if self.ll_feat_y_com[i][2] == com]

            ll_classified = []
            for i in range(len(one_com_features)):
                ll_classified.append(
                    classify(one_com_features[i], self.my_tree))

            ll_keys_classified = [list(ll_classified[i].keys())[0]
                                  for i in range(len(ll_classified))]

            best_name = Counter(ll_keys_classified).most_common(1)[0][0]

            self.y_hat_dict[best_name] = self.y_hat_dict[best_name] + \
                self.y_com_dict[com]

            ll_best_name.append(best_name)

    def build_empty_dict(self):
        """ ."""
        ll_communities = list(self.resultDic[0].ComDict.keys())
        length_df = np.cumsum([len(self.resultDic[i].signal)
                               for i in self.resultDic.keys()])[-1]

        # Community to time series appliances channel
        y_com_dict = {}
        for com in ll_communities:
            y_com_dict[com] = np.concatenate(
                [self.resultDic[i].ComDict[com].sum(axis=1) for i in self.resultDic.keys()])

        # Ground truth appliances channel
        y_truth_dict = {}
        for ll in self.ll_columns:
            y_truth_dict[ll] = np.concatenate(
                [self.resultDic[i].dataframe[ll].values for i in self.resultDic.keys()])

        # Predictited appliances channel
        y_hat_dict = {}
        for ll in self.ll_columns:
            y_hat_dict[ll] = np.zeros(length_df)

        return y_com_dict, y_truth_dict, y_hat_dict

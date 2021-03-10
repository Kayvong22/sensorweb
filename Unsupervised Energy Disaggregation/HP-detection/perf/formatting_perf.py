from perf.perf_metrics import *

# TODO explanation of the class

class format_perf(object):
    """docstring for ."""

    # ll_perf_metrics = [est_acc, mae, rmse,
    #                    recall, precision, f1_score, accuracy]

    ll_perf_metrics = [est_acc, f1_score, precision, recall, accuracy]

    def __init__(self, resultDic, y_hat_dict, y_truth_dict):
        self.resultDic = resultDic
        self.y_hat_dict = y_hat_dict
        self.y_truth_dict = y_truth_dict

    def individual_appliances_perf(self):
        self.global_performance = {}

        for i, func in enumerate(self.ll_perf_metrics):
            self.global_performance[i] = {}

            for appliance in self.y_truth_dict.keys():
                y_hat = self.y_hat_dict[appliance]
                y_truth = self.y_truth_dict[appliance]

                self.global_performance[i][appliance] = func(y_hat, y_truth)

    def overall_dissag_perf(self):

        ll_keys = list(self.resultDic.keys())
        signal = [self.resultDic[i].signal for i in ll_keys]
        signal = np.concatenate(signal)
        recover = [self.resultDic[i].RecSignal for i in ll_keys]
        recover = np.concatenate(recover)

        #for i, func in enumerate(self.ll_perf_metrics):
        #    self.global_performance[i]['Across all appliances'] = func(signal, recover)
        self.global_performance[0]['Across all appliances'] = est_acc(signal, recover)
        for i in range(1,5):
            self.global_performance[i]['Across all appliances'] =np.mean([self.global_performance[i][j]for j in self.global_performance[i].keys()])

    def output_performance(self):

        self.individual_appliances_perf()
        self.overall_dissag_perf()

        df_global_performance = pd.DataFrame(
            [self.global_performance[i] for i in range(len(self.ll_perf_metrics))])

        # df_global_performance.index = ['Est. Accuracy', 'MAE',
        #                                'RMSE', 'Recall', 'Precision', 'f1-score', 'Accuracy']

        df_global_performance.index = ['Est. Accuracy', 'f1-score', 'Precision', 'Recall', 'Accuracy']


        return df_global_performance

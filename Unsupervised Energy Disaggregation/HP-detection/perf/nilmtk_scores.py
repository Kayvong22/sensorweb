def f1_score(predictions, ground_truth):
    '''Compute F1 scores.
    .. math::
        F_{score}^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}
    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup
    Returns
    -------
    f1_scores : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the F1 score for that appliance.  If there are multiple
        chunks then the value is the weighted mean of the F1 score for
        each chunk.
    '''
    # If we import sklearn at top of file then sphinx breaks.
    from sklearn.metrics import f1_score as sklearn_f1_score

    # sklearn produces lots of DepreciationWarnings with PyTables
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    f1_scores = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        scores_for_meter = pd.DataFrame(columns=['score', 'num_samples'])
        aligned_meters = align_two_meters(
            pred_meter, ground_truth_meter, 'when_on')
        for aligned_states_chunk in aligned_meters:
            aligned_states_chunk.dropna(inplace=True)
            aligned_states_chunk = aligned_states_chunk.astype(int)
            score = sklearn_f1_score(aligned_states_chunk.iloc[:, 0],
                                     aligned_states_chunk.iloc[:, 1])
            scores_for_meter = scores_for_meter.append(
                {'score': score, 'num_samples': len(aligned_states_chunk)},
                ignore_index=True)

        # Calculate weighted mean
        num_samples = scores_for_meter['num_samples'].sum()
        if num_samples > 0:
            scores_for_meter['proportion'] = (
                scores_for_meter['num_samples'] / num_samples)
            avg_score = (
                scores_for_meter['score'] * scores_for_meter['proportion']
            ).sum()
        else:
            warn("No aligned samples when calculating F1-score for prediction"
                 " meter {} and ground truth meter {}."
                 .format(pred_meter, ground_truth_meter))
            avg_score = np.NaN
        f1_scores[pred_meter.instance()] = avg_score

    return pd.Series(f1_scores)


##### FUNCTIONS BELOW THIS LINE HAVE NOT YET BEEN CONVERTED TO NILMTK v0.2 #####

def tp_fp_fn_tn(predicted_states, ground_truth_states):
    '''Compute counts of True Positives, False Positives, False Negatives, True Negatives
    
    .. math::
        TP^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = on, \\hat{x}^{(n)}_t = on \\right )
        
        FP^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = off, \\hat{x}^{(n)}_t = on \\right )
        
        FN^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = on, \\hat{x}^{(n)}_t = off \\right )
        
        TN^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = off, \\hat{x}^{(n)}_t = off \\right )
    Parameters
    ----------
    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}
    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}
    Returns
    -------
    numpy array where columns represent appliances and rows represent: [TP, FP, FN, TN]
    '''
    # assumes state 0 = off, all other states = on
    predicted_states_on = predicted_states > 0
    ground_truth_states_on = ground_truth_states > 0
    tp = np.sum(np.logical_and(predicted_states_on == True,
                ground_truth_states_on == True), axis=0)
    fp = np.sum(np.logical_and(predicted_states_on == True,
                ground_truth_states_on == False), axis=0)
    fn = np.sum(np.logical_and(predicted_states_on == False,
                ground_truth_states_on == True), axis=0)
    tn = np.sum(np.logical_and(predicted_states_on == False,
                ground_truth_states_on == False), axis=0)
    return np.array([tp, fp, fn, tn]).astype(float)
def tpr_fpr(predicted_states, ground_truth_states):
    '''Compute True Positive Rate and False Negative Rate
    
    .. math::
        TPR^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}
        
        FPR^{(n)} = \\frac{FP}{\\left ( FP + TN \\right )}
    Parameters
    ----------
    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}
    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}
    Returns
    -------
    numpy array where columns represent appliances and rows represent: [TPR, FPR]
    '''
    tfpn = tp_fp_fn_tn(predicted_states, ground_truth_states)
    tpr = tfpn[0, :] / (tfpn[0, :] + tfpn[2, :])
    fpr = tfpn[1, :] / (tfpn[1, :] + tfpn[3, :])
    return np.array([tpr, fpr])
def precision_recall(predicted_states, ground_truth_states):
    '''Compute Precision and Recall
    
    .. math::
        Precision^{(n)} = \\frac{TP}{\\left ( TP + FP \\right )}
        
        Recall^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}
    Parameters
    ----------
    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}
    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}
    Returns
    -------
    numpy array where columns represent appliances and rows represent: [Precision, Recall]
    '''
    tfpn = tp_fp_fn_tn(predicted_states, ground_truth_states)
    prec = tfpn[0, :] / (tfpn[0, :] + tfpn[1, :])
    rec = tfpn[0, :] / (tfpn[0, :] + tfpn[2, :])
    return np.array([prec, rec])
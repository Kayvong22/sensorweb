import pandas
import numpy
import numpy as np
from lab.snippet4lab import *
from scipy.integrate import simps

def extract_feature(signal, samples_per_min):
    """Extract the feature from the signal input (np.array, pd.ts or pd.df).

    Output: is a list() of features.

    Process: ifelse matching the type of the input.
    """
    # Check the correct type
    assert isinstance(
        signal, (pandas.core.series.Series,
                 pandas.core.frame.DataFrame,
                 numpy.ndarray)),   "Wrong input type, signal should be pandas.series OR pandas.df OR numpy.array"

    if isinstance(signal, pandas.core.frame.DataFrame):

        
        all_name_appl=[]
        for name_appl in signal.columns:

            y = signal[name_appl].values

            snip = snippets4lab(y, deletion_zero_snippets=False)
            snip_dict = snip.dict_snippets_without_zeros()

            for j,i in enumerate(snip_dict.keys()):
                fx = snip_dict[i][snip_dict[i]!=0]
                #ll_appl = []

                # magnitude
                x_overline = round(np.mean(fx),2)
                #ll_appl.append(x_overline)

                # Wh of the event (energy)
                
                #area = np.sum(fx) * (len(fx) - 1) / len(fx)
                #energy_event = round(area * 1 / (60/samples_per_min),2)
                energy_event =round(simps(fx,dx=6),2)
                #ll_appl.append(energy_event)

                peak_event =round(np.max(fx),2)
                #ll_appl.append(peak_event)

                bottom_event =round(np.min(fx),2)
                #ll_appl.append(bottom_event)

                median_event =round(np.median(fx),2)
                #ll_appl.append(median_event)
                # frequency
                # ll_appl.append(ll_freq4input[j])

                operation_time=len(fx)
                #ll_appl.append(operation_time)

                variance_event=np.var(fx)
                #ll_appl.append(variance_event)
                # name of the appliance
                ll_appl=np.array((x_overline,energy_event,median_event, variance_event, peak_event, bottom_event, operation_time))
                all_name_appl.append(name_appl)
                try:
                    ll_feat_appl=np.vstack((ll_feat_appl,ll_appl))
                except NameError:
                    ll_feat_appl=np.array(ll_appl)

    elif isinstance(signal, numpy.ndarray):

        y = signal
        #ll_feat_appl = []
        #all_name_appl=[]
        snip = snippets4lab(y, deletion_zero_snippets=False)
        snip_dict = snip.dict_snippets_without_zeros()

        for j, i in enumerate(snip_dict.keys()):


            fx = snip_dict[i][snip_dict[i]!=0]
            #ll_appl = []

            # magnitude
            x_overline = round(np.mean(fx),2)
            #ll_appl.append(x_overline)

            # Wh of the event (energy)
            
            #area = np.sum(fx) * (len(fx) - 1) / len(fx)
            #energy_event = round(area * 1 / (60/samples_per_min),2)
            energy_event =round(simps(fx,dx=6),2)
            #ll_appl.append(energy_event)

            peak_event =round(np.max(fx),2)
            #ll_appl.append(peak_event)

            bottom_event =round(np.min(fx),2)
            #ll_appl.append(bottom_event)

            median_event =round(np.median(fx),2)
            #ll_appl.append(median_event)
            # frequency
            # ll_appl.append(ll_freq4input[j])

            operation_time=len(fx)
            #ll_appl.append(operation_time)

            variance_event=np.var(fx)
            #ll_appl.append(variance_event)
            # name of the appliance
            ll_appl=np.array((x_overline,energy_event,median_event, variance_event, peak_event, bottom_event, operation_time))
           #all_name_appl.append(name_appl)
            try:
                ll_feat_appl=np.vstack((ll_feat_appl,ll_appl))
            except NameError:
                ll_feat_appl=np.array(ll_appl)

            # ll_appl = []

            # # magnitude
            # x_overline = np.mean(snip_dict[i][snip_dict[i] > 0])
            # ll_appl.append(x_overline)

            # # Wh of the event (energy)
            # fx = snip_dict[i]
            # area = np.sum(fx) * (len(fx) - 1) / len(fx)
            # energy_event = area * 1 / (60/samples_per_min)
            # ll_appl.append(energy_event)

            # # frequency
            # # ll_appl.append(ll_freq4input[j])

            # ll_feat_appl.append(ll_appl)

    elif isinstance(signal, pandas.core.series.Series):
        self.signal = signal.values

        y = signal.values
        ll_feat_appl = []

        snip = snippets4lab(y, deletion_zero_snippets=False)
        snip_dict = snip.dict_snippets_without_zeros()

        for j, i in enumerate(snip_dict.keys()):

            ll_appl = []

            # magnitude
            x_overline = np.mean(snip_dict[i][snip_dict[i] > 0])
            ll_appl.append(x_overline)

            # Wh of the event (energy)
            fx = snip_dict[i]
            area = np.sum(fx) * (len(fx) - 1) / len(fx)
            energy_event = area * 1 / (60/samples_per_min)
            ll_appl.append(energy_event)

            # frequency
            # ll_appl.append(ll_freq4input[j])

            ll_feat_appl.append(ll_appl)

    return (ll_feat_appl)#,np.array(all_name_appl))

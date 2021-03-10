from tools.handling import *
from experiments.test_ukdale.params import *
from lab.features_extraction import *
from sklearn.ensemble import RandomForestClassifier
from omp.omp import *
from omp.dictionary import *
from omp.snippet4omp import *
from multiprocessing import Pool
import datetime
def OMP_pool(SE):
	#print(SE[0])
	array_TEMP = TEMP[SE[1]:SE[2]]
	TEMP_selectDF = TEMP_totDF[SE[1]:SE[2]]

	limX = array_TEMP.shape[0]
	Xcut = X[0:limX, :]
	try:
		result = OMP(x=array_TEMP, Dictionary=Xcut, dataframe=TEMP_selectDF,
	 	S=200,
	 	tol=omp_tolerance,
	 	threshold_min_power=50)
	except ValueError:
		result=None
	return(list((SE[0],result)))

# --------------------- Classifer Training ---------------------------------- #




with open(path2store + 'dictdf_appl_training_' + name_store_dataset + '.pickle', 'rb') as handle:
	dictdf_appl_training = pickle.load(handle)

#dictdf_appl_training = dictdf_appl_training.drop(
#columns=['TotalPower_Appliances'])



snippet = snippet4omp(dictdf_appl_training['TotalPower_Appliances'])
snippet.negl_small_power(threshold=70)
snippet.encoding_start_end(threshold=5)
snippet.restrict_start_end(threshold=1000)
snippet.del_snippet_zero_power()


start_omp_loop = snippet.start_omp_loop
end_omp_loop = snippet.end_omp_loop

# --------------------- OMP iteration ---------------------


global X
global TEMP
global TEMP_totDF

try:
	X = np.load(path2store + 'X.npy')
except FileNotFoundError:
		import omp.test_dictionary

limX = X.shape[0]


start_end_omp_loop = np.transpose(np.stack((range(len(snippet.start_omp_loop)),snippet.start_omp_loop,snippet.end_omp_loop)))

ll_appliances = list(dictdf_appl_training.columns)[:-1]
print(datetime.datetime.now())

for appliance in ll_appliances:


	# signal = signal[0:10000]
	# plt.plot(signal[0:10000])

	# --------------------- Preset params --------------------- #
	# Compute the start and end for the snippets
	# snippet = snippet4omp(df_appl)
	
	TEMP = np.asarray(dictdf_appl_training[appliance])
	TEMP_totDF = dictdf_appl_training

	resultDic = {}
	pool=Pool(processes=7) 
	results=pool.map(OMP_pool,start_end_omp_loop)
	#for i in range(len(start_end_omp_loop)):
	#	resultDic[i]=OMP_pool(start_end_omp_loop[i])
	for i in results:
		resultDic[i[0]]=i[1]


	if resultDic:
		print(appliance)
		print(len(resultDic))
		print(datetime.datetime.now())
	output=open(path2store+'result_ukdale_1min_training_{}_omp{}.pickle'.format(appliance, omp_tolerance), 'wb')
	pickle.dump(resultDic, output)
	output.close()



import random
output=open(path2store+'result_ukdale_1min_training_{}_omp{}.pickle'.format(ll_appliances[0], omp_tolerance), 'rb')
data=pickle.load(output)
output.close()

data_clean = {k:v for k,v in data.items() if v is not None} 

samples=np.random.randint(0,7924,10)

[list(data_clean.keys())[j] for j in samples]
to_plot=Kcoef_hstack_set(1000,data_clean)

for i in start_end_omp_loop:
	if 
	plt.plot(toplo,'-b',alpha=0.5)

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# types: 0 = F1-Micro, 1 = F1-Macro, 2 = Simple Accuracy
def performance_calculator(types, target, predict):
	totaldata = target.shape[0]
	if types == 2:
		totaltrue = 0.0
		for i in xrange(totaldata):
			if target[i] == predict[i]:
				totaltrue += 1
		return totaltrue / totaldata
	else:
		list_class = np.unique(target)
		confmatrix = np.array([np.array([0.0 for i in xrange(4)]) for i in xrange(list_class.shape[0])])
		for i in xrange(totaldata):
			confmatrix += [0,0,0,1]
			chosen = np.where(list_class==target[i])
			if target[i] == predict[i]:
				confmatrix[chosen] += [1,0,0,-1]
			else:
				confmatrix[chosen] += [0,0,1,-1]
				confmatrix[np.where(list_class==predict[i])] += [0,1,0,-1]
		if types == 0:
			precision = confmatrix[:,0].sum() / (confmatrix[:,0].sum() + confmatrix[:,1].sum())
			recall =  confmatrix[:,0].sum() / (confmatrix[:,0].sum() + confmatrix[:,2].sum())
		else:
			precision,recall = [], []
			for i in confmatrix:
				precision.append(i[0]/(i[0]+i[1]))
				recall.append(i[0]/(i[0]+i[2]))
			precision = np.average(precision)
			recall = np.average(recall)
		return 2 * (precision * recall)/(precision + recall)

def visualize_data(data,name,decisionboundary=False,boundaryresult=None,modelfunc=None):
	fig, x = plt.subplots()
	fig.suptitle(name)
	list_class = np.unique(data[:,2].astype('i'))
	colors = ['r', 'g', 'b', 'y', 'k', 'c']
	marker = ['o', 's', '*']
	mcounter, ccounter = 0,0

	if (decisionboundary):
		h = 0.1
		maxmin = np.array([data.max(axis=0)+1,data.min(axis=0)-1])
		xx, yy = np.meshgrid(np.arange(maxmin[1,0], maxmin[0,0], h),
		                     np.arange(maxmin[1,1], maxmin[0,1], h))
		Z = np.array(modelfunc[1](np.c_[xx.ravel(), yy.ravel()],modelfunc[0]))

		Z = Z.reshape(xx.shape)
		x.contourf(xx, yy, Z, cmap=plt.cm.Paired)
		x.axis('off')

	for i in list_class:
		data_class = data[data[:,2] == i]
		x.scatter(data_class[:,0], data_class[:,1], c=colors[(ccounter) % 6], marker=marker[(mcounter) % 3])
		ccounter += 1
		if(ccounter % 3 == 0): mcounter += 1


def naive_learn(data, targetcol):
	list_class = np.unique(data[:,targetcol].astype('i'))
	dataperclass,prior,mean,std = {},{},{},{}
	likelihood = []

	for i in list_class:
		dataperclass[i] = data[data[:,targetcol] == i]
		prior[i] = dataperclass[i].shape[0] / (data.shape[0] + 0.0)
		mean[i] = [np.mean(dataperclass[i][:,j]) for j in xrange(dataperclass[i].shape[1] - 1)]
		std[i] = [np.std(dataperclass[i][:,j]) for j in xrange(dataperclass[i].shape[1] - 1)]

	return prior,mean,std

def naive_bayes(data, model):
	prior, mean, std = model
	classification_result = []
	gausbayes = lambda m,sd,x: (1/(sd*np.sqrt(2*np.pi)))*np.exp(-(((x-m)**2)/(2*sd**2)))
	likelihood = []

	for values in data:
		likelihood.append({i: [gausbayes(mean[i][j],std[i][j],values[j]) for j in xrange(len(mean[i]))] for i in prior.keys()})

	for x in xrange(data.shape[0]):
		posterior = {}
		for i in prior.keys():
			posterior[i] = np.log(prior[i]) + np.sum(np.log(likelihood[x][i]))
		classification_result.append(max(posterior, key=posterior.get))
	return classification_result

def ann_learn(data, targetcol):
	list_class = np.unique(data[:,targetcol].astype('i'))
	# configurasi
	hidden_neuron = [4] #panjang array = jumlah layer
	output_neuron = list_class.shape[0]
	lr = 0.01
	eppoch = 100
	msetreshold = 10**-2

	# inisialisasi
	hinput = [data.shape[1] - 1 if i == 0 else hidden_neuron[i-1] for i in range(len(hidden_neuron))]
	whidden = np.array([[[rnd.random() for k in xrange(data.shape[1] - 1) if i==0 else rnd.random() for k in xrange(hidden_neuron[i-1])] for j in xrange(hidden_neuron[i])] for i in xrange(len(hidden_neuron))])
	woutput = np.array([[rnd.random() for j in xrange(hidden_neuron[-1])] for i in xrange(output_neuron)])


	print whidden
	print woutput
	# for i in xrange(eppoch):




data = np.genfromtxt('R15.csv',delimiter=',')
model = ann_learn(data, 2)
# classification = naive_bayes(data, model)
# dataclassification = np.copy(data)
# dataclassification[:,-1] = classification
# visualize_data(data,'PathBased Plot')
# visualize_data(dataclassification,'PathBased Naive Bayes')
# print performance_calculator(0, data[:,-1], classification)
# visualize_data(data,'PathBased DecisionBoundary with Naive Bayes',True,classification,[model,naive_bayes])
# plt.show()
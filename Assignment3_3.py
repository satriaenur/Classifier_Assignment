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

def ann_learn(data, targetcol, bias = False):
	list_class = np.unique(data[:,targetcol].astype('i'))
	# configurasi
	hidden_neuron = [4,6] #panjang array = jumlah layer
	output_neuron = list_class.shape[0]
	lr = 0.001
	eppoch = 5000
	# msetreshold = 10**-2

	# inisialisasi
	targetjob = [[1 if (j+1)==list_class[i] else 0 for j in xrange(list_class.shape[0])] for i in range(list_class.shape[0])]
	hinput = [data.shape[1] - 1 if i == 0 else hidden_neuron[i-1] for i in range(len(hidden_neuron))]
	whidden = np.array([np.random.uniform(-1,1,(hidden_neuron[i],hinput[i])) for i in xrange(len(hidden_neuron))])
	woutput = np.random.uniform(-1,1,(output_neuron,hidden_neuron[-1]))
	if (bias):
		bhidden = [[rnd.random() for j in xrange(hidden_neuron[i])]for i in xrange(len(hidden_neuron))]
		boutput = [rnd.random() for i in xrange(output_neuron)]
	MSE = 0
	# start learning
	for i in xrange(eppoch):
		for cd in data:
			# Forward
			p = cd[:-1]
			target = cd[-1].astype('i')
			A1 = []
			for hlayer in range(len(whidden)):
				v = np.sum(p * whidden[hlayer], axis=1)
				p = 1/(1 + np.exp(-v))
				A1.append(p)

			A1 = np.array(A1)

			v = np.sum(p * woutput,axis=1)
			A2 = np.array(1/(1 + np.exp(-v)))

			# count error
			e = targetjob[target-1] - A2
			MSE += np.sum(e)**2

			# Backward
			D2 = A2*(1 - A2)*e
			D1 = [D2.dot(woutput) * A1[-1]*(1-A1[-1])]
			for hlayer in range(len(hidden_neuron)-1):
				D1.append(D1[hlayer].dot(whidden[-(hlayer+1)]) * A1[-(hlayer+2)]*(1-A2[-(hlayer+2)]))
				whidden[-(hlayer+1)] += lr*np.outer(D1[hlayer], A1[-(hlayer+2)])
			
			woutput += lr*np.outer(D2,A1[-1])


		MSE = MSE/data.shape[0]
		print MSE
	return whidden,woutput

def ann_test(data, model):
	whidden,woutput = model[:2]
	if(len(model) > 2):
		b1,b2 = model[2:]

	predict = []
	for cd in data:
		p = np.copy(cd)
		A1 = []
		for hlayer in range(len(whidden)):
			v = np.sum(p * whidden[hlayer], axis=1)
			p = 1/(1 + np.exp(-v))
			A1.append(p)

		A1 = np.array(A1)

		v = np.sum(A1[-1] * woutput,axis=1)
		A2 = np.array(1/(1 + np.exp(-v)))
		predict.append(np.argmax(A2) + 1)
	return predict

data = np.genfromtxt('R15.csv',delimiter=',')
model = ann_learn(data, 2)
classification = ann_test(data[:,:-1], model)
# classification = naive_bayes(data, model)
dataclassification = np.copy(data)
dataclassification[:,-1] = classification
visualize_data(data,'PathBased Plot')
visualize_data(dataclassification,'PathBased Naive Bayes')
print performance_calculator(0, data[:,-1], classification)
visualize_data(data,'PathBased DecisionBoundary with Naive Bayes',True,classification,[model,ann_test])
plt.show()
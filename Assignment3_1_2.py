import numpy as np
import matplotlib.pyplot as plt

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
				confmatrix[chosen] += [0,1,0,-1]
				confmatrix[np.where(list_class==predict[i])] += [0,0,1,-1]
		if types == 0:
			confmatrix = confmatrix.sum(axis=0)
			precision = confmatrix[0] / (confmatrix[0] + confmatrix[2])
			recall = confmatrix[0] / (confmatrix[0] + confmatrix[1])
		else:
			precision = (confmatrix[:,0] / (confmatrix[:,0] + confmatrix[:,2])).sum()/confmatrix.shape[0]
			recall =  (confmatrix[:,0] / (confmatrix[:,0] + confmatrix[:,1])).sum()/confmatrix.shape[0]
		return 2*(precision * recall)/(precision + recall)

def visualize_data(data,decisionboundary=False):
	figure = plt.figure()
	list_class = np.unique(data[:,2].astype('i'))
	colors = ['r', 'g', 'b', 'y', 'k', 'c']
	marker = ['o', 's', '*']
	mcounter, ccounter = 0,0
	for i in list_class:
		data_class = data[data[:,2] == i]
		plt.scatter(data_class[:,0], data_class[:,1], c=colors[(ccounter) % 6], marker=marker[(mcounter) % 3])
		ccounter += 1
		if(ccounter % 3 == 0): mcounter += 1
	if (decisionboundary):
		boundary = np.array([data.max(axis=0),data.min(axis=0)])
		plt.plot([boundary[0,0],boundary[1,0]],[boundary[0,1],boundary[1,1]])

class naive_bayes:
	def __init__(self, data, targetcol):
		self.data = data
		self.targetcol = targetcol
		self.prior = {}
		self.mean = {}
		self.variance = {}
		self.learning()
		# self.predict()

	def learning(self):
		self.list_class = np.unique(data[:,self.targetcol].astype('i'))
		dataperclass = {}
		for i in self.list_class:
			dataperclass[i] = data[data[:,2] == i]
			self.prior[i] = dataperclass[i].shape[0] / (data.shape[0] + 0.0)
			self.mean[i] = [np.mean(dataperclass[i][:,j]) for j in xrange(dataperclass[i].shape[1] - 1)]
			self.variance[i] = [np.var(dataperclass[i][:,j],ddof=1) for j in xrange(dataperclass[i].shape[1] - 1)]

	def predict(self,values):
		predict_result = {}
		for i in self.list_class:
			predict_result[i] = np.log(self.prior[i]) + np.sum([np.log(1/(np.sqrt(2*np.pi*np.sqrt(self.variance[i][j])))*np.exp(((values[j]-self.mean[i][j])**2)/(2*self.variance[i][j]))) for j in xrange(values.shape[0]-1)])
		print predict_result, max(predict_result, key=predict_result.get)

data = np.genfromtxt('jain.csv',delimiter=',')
nb = naive_bayes(data, 2)
print data[1]
nb.predict(data[1])

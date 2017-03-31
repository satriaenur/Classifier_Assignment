import numpy as np

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



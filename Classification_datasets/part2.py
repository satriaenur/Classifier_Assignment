import numpy as np
import matplotlib.pyplot as plt
import random

data = np.genfromtxt("jain.csv", dtype= float, delimiter=",")


def confussionMatrix(data, predict, classes):
	x = np.array([0.0, 0.0, 0.0, 0.0]) #[TP, FP, TN, FN]
	for i in range(len(data)):
		if  (data[i,2] == predict[i,2]):
			if (predict[i,2] == classes):
				x += [1,0,0,0]
			else:
				x += [0,0,1,0]
		elif (predict[i,2] != data[i,2]):
			if(predict[i,2] == classes):
				x += [0,1,0,0]
			else:
				x += [0,0,0,1]
	return x

def classifierPerformance(type, conf1, conf2):
	if type == 'macro':
		precission1 = conf1[0]/conf1[0]+conf1[1]
		precission2 = conf2[0]/conf2[0]+conf2[1]
		finprecission = (precission1+precission2)/2
		recall1 = conf1[0]/conf1[0]+conf1[3]
		recall2 = conf2[0]/conf2[0]+conf2[3]
		finrec = (recall1+recall2)/2
		return (2*(finprecission*finrec))/(finprecission+finrec)
	elif type == 'micro':
		confMatrix = conf1+conf2
		precission = confMatrix[0]/(confMatrix[0]+confMatrix[1])
		recall = confMatrix[0]/(confMatrix[0]+confMatrix[3])
		return (2*(precission*recall))/(precission+recall)
	elif type == 'simple accuracy':
		confMatrix = conf1+conf2
		return (confMatrix[0]+confMatrix[2])/sum(confMatrix)

def visualizeScatterPlot(data,name):
	fig, plot = plt.subplots()
	fig.suptitle(name)
	for i in data:
		if (i[2] == 1):
			plot.scatter(i[0], i[1], marker='x', color='b')
		else:
			plot.scatter(i[0], i[1], marker='o', color='r')

def prior(data):
	prior1, prior2 = 0.0, 0.0
	for i in data:
		if(i[2] ==1):
			prior1 += 1
		else:
			prior2 += 1
	prior1 = prior1/(len(data))
	prior2 = prior2/(len(data))
	x1, x2= [], []
	y1, y2= [], []
	for i in data:
		if i[2] == 1:
			x1.append(i[0])
			y1.append(i[1])
		else:
			x2.append(i[0])
			y2.append(i[1])
	sd = [[np.std(x1),np.std(x2)],[np.std(y1),np.std(y2)]]
	avg = [[np.mean(x1),np.mean(x2)],[np.mean(y1),np.mean(y2)]]
	return prior1, prior2, sd, avg

def classifier(data, prior1, prior2, sd, avg):
	result = []
	posterior = []
	likelihood = []

	for i in data:
		class1x1 = (1/(sd[0][0]*(np.sqrt(2*(np.pi)))))*(np.exp(-(np.power((i[0]-avg[0][0]),2))/(2*(sd[0][0]*sd[0][0]))))
		class1x2 = (1/(sd[1][0]*(np.sqrt(2*(np.pi)))))*(np.exp(-(np.power((i[1]-avg[1][0]),2))/(2*(sd[1][0]*sd[1][0]))))
		class2x1 = (1/(sd[0][1]*(np.sqrt(2*(np.pi)))))*(np.exp(-(np.power((i[0]-avg[0][1]),2))/(2*(sd[0][1]*sd[0][1]))))
		class2x2 = (1/(sd[1][1]*(np.sqrt(2*(np.pi)))))*(np.exp(-(np.power((i[1]-avg[1][1]),2))/(2*(sd[1][1]*sd[1][1]))))
		likelihood.append([class1x1,class1x2,class2x1,class2x2])

	for i in likelihood:
		post1 = np.log(prior1)+np.log(i[0])+np.log(i[1])
		post2 = np.log(prior2)+np.log(i[2])+np.log(i[3])
		posterior.append([post1,post2])

	for i in posterior:
		if (i[0]>i[1]):
			result.append(1)
		else:
			result.append(2)
	return result

def decissionBoundary(data, name, result, model):
	h = 0.1
	fig, plot = plt.subplots()
	fig.suptitle(name)

	x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
	y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = np.array(model[4](np.c_[xx.ravel(), yy.ravel()],model[0],model[1],model[2],model[3]))
	Z = Z.reshape(xx.shape)

	plot.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
	plot.axis('off')

	for i in data:
		if (i[2] == 1):
			plot.scatter(i[0], i[1], marker='x', color='b')
		else:
			plot.scatter(i[0], i[1], marker='o', color='r')

def aNN(data):
	totalData = len(data)
	dimension = data.shape[1]-1
	outputNeuron = 2
	hideNeuron = 6
	learnRate = 0.001
	epoch = 1000
	maxMSE = 0.000000001

	W1 = []
	B1 = []
	for i in range(hideNeuron):
		w = []
		for j in range(dimension):
			w.append(random.uniform(-1.0, 1.0))
		W1.append(w)
		B1.append(random.uniform(-1.0, 1.0))
	W1=np.array(W1)

	W2 = []
	B2 = []
	for i in range(outputNeuron):
		w = []
		for j in range(hideNeuron):
			w.append(random.uniform(-1.0, 1.0))
		W2.append(w)
		B2.append(random.uniform(-1.0, 1.0))
	W2=np.array(W2)
	MSE = 0
	for x in range(epoch):
		for i in data:
			v1 = np.sum(i[:-1] * W1,axis=1)+np.array(B1)
			A1 = 1/(1+np.exp(-(2*v1)))
			v2 = np.sum(W2*A1,axis=1)+np.array(B2)
			A2 =  1/(1+np.exp(-(2*v2)))
			if i[2] == 1:
				output = [3,0]
			else:
				output = [0,3]

			error = output - A2
			MSE += (sum(error)**2)

			#BACK PROPAGATION
			D2 = A2*(1-A2)*error
			D1 = A1*(1-A1)*np.sum(W2.T*D2)
			dW2 = learnRate*np.outer(D2,A1)
			dB2 = learnRate*D2
			dW1 = learnRate*np.outer(D1,i[:-1])
			dB1 = learnRate*D1

			W1 += np.array(dW1)
			W2 += np.array(dW2)
			B1 += np.array(dB1)
			B2 += np.array(dB2)
		MSE = MSE/data.shape[0]
		print x,MSE
		if(MSE<=maxMSE): break
	return W1,W2,B1,B2

def annClassifier(data,W1,W2,B1,B2):
	predict = []
	for i in data:
			v1 = np.sum(i * W1,axis=1)+np.array(B1)
			A1 = 1/(1+np.exp(-(2*v1)))
			v2 = np.sum(W2*A1,axis=1)+np.array(B2)
			A2 =  1/(1+np.exp(-(2*v2)))
			idxmax = np.argmax(A2)
			predict.append(idxmax+1)
	return predict

#NB : Jika ingin melihat hasil visualisasi tinggal uncomment saja untuk setiap visualisasi yang diinginkan
a,b,c,d = prior(data)
f = classifier(data,a,b,c,d)
databaru = np.copy(data)
databaru[:,-1] = f
#visualizeScatterPlot(data,'Jain')
#visualizeScatterPlot(databaru,'Jain Naive Bayes')
#decissionBoundary(data, 'Decision Boundary Jain Naive Bayes', f, [a,b,c,d,classifier])
# print data.shape[1]-1
a,b,c,d = aNN(data)
f = annClassifier(data[:,:-1],a,b,c,d)
databaru = np.copy(data)
databaru[:,-1] = f
conf1= confussionMatrix(data, databaru, 1)
conf2= confussionMatrix(data, databaru, 2)

print "Nilai F1 Micro Average adalah =", classifierPerformance('micro', conf1, conf2)*100 ,"%"
visualizeScatterPlot(data,'Jain')
visualizeScatterPlot(databaru,'Jain ANN')
decissionBoundary(data, 'Decision Boundary Jain ANN', f, [a,b,c,d,annClassifier])
plt.show()

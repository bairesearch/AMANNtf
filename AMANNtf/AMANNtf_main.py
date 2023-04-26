"""AMANNtf_main.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
	
# Usage:
source activate anntf2
python3 AMANNtf_main.py

# Description:
AMANNtf main - train additive-multiplicative artificial neural network (AMANN)

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

from ANNtf_operations import *
import ANNtf_globalDefs
from numpy import random
import ANNtf_loadDataset

#select algorithm:
algorithm = "AMANN"	#additive-multiplicative artificial neural network

suppressGradientDoNotExistForVariablesWarnings = True

costCrossEntropyWithLogits = False
if(algorithm == "AMANN"):
	import AMANNtf_algorithmAMANN as AMANNtf_algorithm
	
#learningRate, trainingSteps, batchSize, displayStep, numEpochs = -1

#performance enhancements for development environment only: 
debugUseSmallPOStagSequenceDataset = True	#def:False	#switch increases performance during development	#eg data-POStagSentence-smallBackup
useSmallSentenceLengths = True	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup
trainMultipleFiles = False	#def:True	#switch increases performance during development	#eg data-POStagSentence
trainMultipleNetworks = False	#trial improve classification accuracy by averaging over multiple independently trained networks (test)

if(trainMultipleFiles):
	randomiseFileIndexParse = True
	fileIndexFirst = 0
	if(useSmallSentenceLengths):
		fileIndexLast = 11
	else:
		fileIndexLast = 1202	#defined by wiki database extraction size
else:
	randomiseFileIndexParse = False
				
#loadDatasetType3 parameters:
#if generatePOSunambiguousInput=True, generate POS unambiguous permutations for every POS ambiguous data example/experience
#if onlyAddPOSunambiguousInputToTrain=True, do not train network with ambiguous POS possibilities
#if generatePOSunambiguousInput=False and onlyAddPOSunambiguousInputToTrain=False, requires simultaneous propagation of different (ambiguous) POS possibilities

numberOfNetworks = 1
trainMultipleNetworks = False

if(algorithm == "AMANN"):
	dataset = "SmallDataset"
	trainMultipleNetworks = False	#default: False		#optional
	if(trainMultipleNetworks):
		numberOfNetworks = 5
				
if(dataset == "SmallDataset"):
	smallDatasetIndex = 0 #default: 0 (New Thyroid)
	#trainMultipleFiles = False	#required
	smallDatasetDefinitionsHeader = {'index':0, 'name':1, 'fileName':2, 'classColumnFirst':3}	
	smallDatasetDefinitions = [
	(0, "New Thyroid", "new-thyroid.data", True),
	(1, "Swedish Auto Insurance", "UNAVAILABLE.txt", False),	#AutoInsurSweden.txt BAD
	(2, "Wine Quality Dataset", "winequality-whiteFormatted.csv", False),
	(3, "Pima Indians Diabetes Dataset", "pima-indians-diabetes.csv", False),
	(4, "Sonar Dataset", "sonar.all-data", False),
	(5, "Banknote Dataset", "data_banknote_authentication.txt", False),
	(6, "Iris Flowers Dataset", "iris.data", False),
	(7, "Abalone Dataset", "UNAVAILABLE", False),	#abaloneFormatted.data BAD
	(8, "Ionosphere Dataset", "ionosphere.data", False),
	(9, "Wheat Seeds Dataset", "seeds_datasetFormatted.txt", False),
	(10, "Boston House Price Dataset", "UNAVAILABLE", False)	#housingFormatted.data BAD
	]
	dataset2FileName = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['fileName']]
	datasetClassColumnFirst = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['classColumnFirst']]
	print("dataset2FileName = ", dataset2FileName)
	print("datasetClassColumnFirst = ", datasetClassColumnFirst)
			
if(debugUseSmallPOStagSequenceDataset):
	dataset1FileNameXstart = "Xdataset1PartSmall"
	dataset1FileNameYstart = "Ydataset1PartSmall"
	dataset3FileNameXstart = "Xdataset3PartSmall"
	dataset4FileNameStart = "Xdataset4PartSmall"
else:
	dataset1FileNameXstart = "Xdataset1Part"
	dataset1FileNameYstart = "Ydataset1Part"
	dataset3FileNameXstart = "Xdataset3Part"
	dataset4FileNameStart = "Xdataset4Part"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "datasetPart"
datasetFileNameEnd = ".dat"
xmlDatasetFileNameEnd = ".xml"


def defineTrainingParameters(dataset, numberOfFeaturesPerWord=None, paddingTagIndex=None):
	return AMANNtf_algorithm.defineTrainingParameters(dataset)

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths=None, numberOfFeaturesPerWord=None):
	return AMANNtf_algorithm.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks)	

def defineNeuralNetworkParameters():
	return AMANNtf_algorithm.defineNeuralNetworkParameters()

#define default forward prop function for backprop weights optimisation;
def neuralNetworkPropagation(x, networkIndex=1):
	return AMANNtf_algorithm.neuralNetworkPropagation(x, networkIndex)
	
#define default forward prop function for test (identical to below);
def neuralNetworkPropagationTest(test_x, networkIndex=1):
	return AMANNtf_algorithm.neuralNetworkPropagation(test_x, networkIndex)

#if(AMANNtf_algorithm.supportMultipleNetworks):
def neuralNetworkPropagationLayer(x, networkIndex, l):
	return AMANNtf_algorithm.neuralNetworkPropagationLayer(x, networkIndex, l)
def neuralNetworkPropagationAllNetworksFinalLayer(x):
	return AMANNtf_algorithm.neuralNetworkPropagationAllNetworksFinalLayer(x)


def trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display):
	
	if(algorithm == "AMANN"):
		executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)

	if(display):
		loss, acc = calculatePropagationLoss(batchX, batchY, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex)
		print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
			
def executeLearningEIANN(x, y, networkIndex):
	#first learning algorithm: perform neuron independence training
	pred = AMANNtf_algorithm.neuralNetworkPropagationEIANNtrain(x, networkIndex)

def executeOptimisation(x, y, datasetNumClasses, numberOfLayers, optimizer, networkIndex=1):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex)
		
	if(algorithm == "AMANN"):
		Wlist = []
		Blist = []
		for l1 in range(1, numberOfLayers+1):
			if(AMANNtf_algorithm.supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						if(l1 < numberOfLayers):
							Wlist.append(AMANNtf_algorithm.Wa[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")])
							Wlist.append(AMANNtf_algorithm.Wm[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wm")])
						else:
							if(l2 == l1-1):	#l1 == numberOfLayers
								Wlist.append(AMANNtf_algorithm.Wa[generateParameterNameNetworkSkipLayers(networkIndex, l1-1, l1, "Wa")])
				Blist.append(AMANNtf_algorithm.Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")])
				if(l1 < numberOfLayers):
					Blist.append(AMANNtf_algorithm.Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")])		
			else:
				if(AMANNtf_algorithm.debugOnlyTrainFinalLayer):
					if(l1 == numberOfLayers):
						Wlist.append(AMANNtf_algorithm.Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")])
						Blist.append(AMANNtf_algorithm.Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")])							
				else:	
					Wlist.append(AMANNtf_algorithm.Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")])
					Blist.append(AMANNtf_algorithm.Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")])
					if(l1 < numberOfLayers):
						Wlist.append(AMANNtf_algorithm.Wm[generateParameterNameNetwork(networkIndex, l1, "Wm")])
						Blist.append(AMANNtf_algorithm.Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")])	
					
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))

def calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex=1):
	acc = 0	#only valid for softmax class targets 
	
	pred = neuralNetworkPropagation(x, networkIndex)
	target = y
	loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)	
	acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 
	#print("x = ", x)
	#print("y = ", y)
	#print("2 loss = ", loss)
	#print("2 acc = ", acc)
			
	return loss, acc



#if(AMANNtf_algorithm.supportMultipleNetworks):

def testBatchAllNetworksFinalLayer(batchX, batchY, datasetNumClasses, numberOfLayers):
	
	AfinalHiddenLayerList = []
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	#print("AfinalHiddenLayerTensor = ", AfinalHiddenLayerTensor)
	
	pred = neuralNetworkPropagationAllNetworksFinalLayer(AfinalHiddenLayerTensor)
	acc = calculateAccuracy(pred, batchY)
	print("Combined network: Test Accuracy: %f" % (acc))
	
def trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display):
	
	AfinalHiddenLayerList = []
	for networkIndex in range(1, numberOfNetworks+1):
		AfinalHiddenLayer = neuralNetworkPropagationLayer(batchX, networkIndex, numberOfLayers-1)
		AfinalHiddenLayerList.append(AfinalHiddenLayer)	
	AfinalHiddenLayerTensor = tf.concat(AfinalHiddenLayerList, axis=1)
	#print("AfinalHiddenLayerTensor = ", AfinalHiddenLayerTensor)
	#print("AfinalHiddenLayerTensor.shape = ", AfinalHiddenLayerTensor.shape)
	
	executeOptimisationAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, optimizer)

	pred = None
	if(display):
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(AfinalHiddenLayerTensor, batchY, datasetNumClasses, costCrossEntropyWithLogits)
		print("Combined network: batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))
						
def executeOptimisationAllNetworksFinalLayer(x, y, datasetNumClasses, optimizer):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits)
		
	Wlist = []
	Blist = []
	Wlist.append(AMANNtf_algorithm.WallNetworksFinalLayer)
	Blist.append(AMANNtf_algorithm.BallNetworksFinalLayer)
	trainableVariables = Wlist + Blist

	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))
			
def calculatePropagationLossAllNetworksFinalLayer(x, y, datasetNumClasses, costCrossEntropyWithLogits):
	acc = 0	#only valid for softmax class targets 
	#print("x = ", x)
	pred = neuralNetworkPropagationAllNetworksFinalLayer(x)
	#print("calculatePropagationLossAllNetworksFinalLayer: pred.shape = ", pred.shape)
	target = y
	loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)	
	acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 

	return loss, acc
	
	
		
def loadDataset(fileIndex):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameStart + fileIndexStr + xmlDatasetFileNameEnd
			
	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf_loadDataset.loadDatasetType4(datasetType4FileName, AEANNsequentialInputTypesMaxLength, useSmallSentenceLengths, AEANNsequentialInputTypeMinWordVectors)

	if(dataset == "wikiXmlDataset"):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp
		


#trainMinimal is minimal template code extracted from train based on trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False;
def trainMinimal():
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		fileIndex = 0
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

		shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
		trainDataIndex = 0

		trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
		trainDataList = []
		trainDataList.append(trainData)
		trainDataListIterators = []
		for trainData in trainDataList:
			trainDataListIterators.append(iter(trainData))
		testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)

		for batchIndex in range(int(trainingSteps)):
			(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
			batchYactual = batchY
					
			display = False
			if(batchIndex % displayStep == 0):
				display = True	
			trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display)

		pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
		print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))

			
#this function can be used to extract a minimal template (does not support algorithm==LREANN);
def train(trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False):
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()

	#configure optional parameters;
	if(trainMultipleNetworks):
		maxNetwork = numberOfNetworks
	else:
		maxNetwork = 1
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	if(greedy):
		maxLayer = numberOfLayers
	else:
		maxLayer = 1
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f
				
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

			shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
			trainDataIndex = 0

			#greedy code;
			for l in range(1, maxLayer+1):
				print("l = ", l)
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
				testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)
				#testBatchX, testBatchY = (test_x, test_y)

				for batchIndex in range(int(trainingSteps)):
					(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
					batchYactual = batchY
					
					for networkIndex in range(1, maxNetwork+1):
						display = False
						#if(l == maxLayer):	#only print accuracy after training final layer
						if(batchIndex % displayStep == 0):
							display = True	
						trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display)
						
					#trainMultipleNetworks code;
					if(trainMultipleNetworks):
						#train combined network final layer
						trainBatchAllNetworksFinalLayer(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, costCrossEntropyWithLogits, display)

				#trainMultipleNetworks code;
				if(trainMultipleNetworks):
					testBatchAllNetworksFinalLayer(testBatchX, testBatchY, datasetNumClasses, numberOfLayers)
				else:
					pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
					if(greedy):
						print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, testBatchY)))
					else:
						print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))


def generateRandomisedIndexArray(indexFirst, indexLast, arraySize=None):
	fileIndexArray = np.arange(indexFirst, indexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	if(arraySize is None):
		np.random.shuffle(fileIndexArray)
		fileIndexRandomArray = fileIndexArray
	else:
		fileIndexRandomArray = random.sample(fileIndexArray.tolist(), arraySize)
	
	print("fileIndexRandomArray = " + str(fileIndexRandomArray))
	return fileIndexRandomArray
	
				
if __name__ == "__main__":
	if(algorithm == "AMANN"):
		if(trainMultipleNetworks):
			train(trainMultipleNetworks=trainMultipleNetworks)
		else:
			trainMinimal()
	else:
		print("main error: algorithm == unknown")
		

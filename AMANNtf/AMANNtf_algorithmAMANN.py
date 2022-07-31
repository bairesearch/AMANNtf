"""AMANNtf_algorithmAMANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see AMANNtf_main.py

# Usage:
see AMANNtf_main.py

# Description:
AMANNtf algorithm AMANN - define additive-multiplicative artificial neural network

specification: additiveMultiplactiveNet-31July2022a.png

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs

debugOnlyTrainFinalLayer = False
debugSingleLayerNetwork = False
debugFastTrain = False

supportMultipleNetworks = False

supportSkipLayers = False


Wa = {}	#additive
Wm = {}	#multiplicative
Ba = {}	#additive
Bm = {}	#multiplicative

if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
if(supportSkipLayers):
	Ztrace = {}
	Atrace = {}

#Network parameters
#interpret n_h[hiddenLayer] (of all actual hidden ie non-input/output layers) as the number of additive or multiplicative units (ie n_h[hiddenLayer] all = n_h[hiddenLayer] additiveOrMultiplactive * 2)
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0

def defineTrainingParameters(dataset):
	global batchSize
	
	learningRate = 0.001
	batchSize = 100
	numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000

	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=False)
	
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	print("numberOfLayers = ", numberOfLayers)
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
			
		for l1 in range(1, numberOfLayers+1):

			if(l1 == numberOfLayers):
				#CHECKTHIS: last layer (fully connected additive layer, no skip layers)
				n_hPreviousLayer = calculateLayerNumHiddenUnits(l1-1)
				Walayer = tf.Variable(randomNormal([n_hPreviousLayer, n_h[l1]]))
				if(supportSkipLayers):
					l2 = l1-1
					Wa[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")] = Walayer
				else:
					Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")] = Walayer
				Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")] = tf.Variable(tf.zeros(n_h[l1]))
			else:			
				if(supportSkipLayers):
					for l2 in range(0, l1):
						if(l2 < l1):
							n_hPreviousLayer = calculateLayerNumHiddenUnits(l2)
							Walayer = tf.Variable(randomNormal([n_hPreviousLayer, n_h[l1]]))
							Wmlayer = tf.Variable(randomNormal([n_hPreviousLayer, n_h[l1]]))
							Wa[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")] = Walayer
							Wm[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wm")] = Wmlayer
							#parameterName = generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")
							#print("parameterName = ", parameterName)
				else:	
					n_hPreviousLayer = calculateLayerNumHiddenUnits(l1-1)
					Walayer = tf.Variable(randomNormal([n_hPreviousLayer, n_h[l1]]))
					Wmlayer = tf.Variable(randomNormal([n_hPreviousLayer, n_h[l1]]))
					Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")] = Walayer
					Wm[generateParameterNameNetwork(networkIndex, l1, "Wm")] = Wmlayer
				Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")] = tf.Variable(tf.zeros(n_h[l1]))
				Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")] = tf.Variable(tf.zeros(n_h[l1]))
							
			if(supportSkipLayers):
				n_hLayer = calculateLayerNumHiddenUnits(l1)
				Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_hLayer], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_hLayer], dtype=tf.dtypes.float32))

	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			n_hPreviousLayer = calculateLayerNumHiddenUnits(numberOfLayers-1)
			WlayerF = randomNormal([n_hPreviousLayer*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer = tf.Variable(WlayerF)
			BlayerF = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer= tf.Variable(BlayerF)	#not currently used

def calculateLayerNumHiddenUnits(l):
	n_hLayerAll = n_h[l]
	if((l > 0) and (l < numberOfLayers)):
		n_hLayerAll = n_hLayerAll*2 #number of units for actual hidden layers: concatenation of additive and multiplicative neurons (x2)
	return n_hLayerAll
											
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationAMANN(x, networkIndex)

def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
	return neuralNetworkPropagationAMANN(x, networkIndex, l)

#if(supportMultipleNetworks):
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred
		
def neuralNetworkPropagationAMANN(x, networkIndex=1, l=None):
			
	#print("numberOfLayers", numberOfLayers)

	if(l == None):
		maxLayer = numberOfLayers
	else:
		maxLayer = l
			
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	
	for l1 in range(1, maxLayer+1):
		print("l1 = " + str(l1))		
		if(l1 == numberOfLayers):
			#CHECKTHIS: last layer (fully connected additive layer, no skip layers)
			if(supportSkipLayers):
				l2 = l1-1
				Walayer = Wa[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")]
			else:
				Walayer = Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")]
			Z = tf.add(tf.matmul(AprevLayer, Walayer), Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")])
			A = activationFunction(Z)
		else:
			if(supportSkipLayers):
				Za = tf.zeros([batchSize, n_h[l1]])
				Zm = tf.zeros([batchSize, n_h[l1]])
				for l2 in range(0, l1):
					AprevLayerA = Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")]
					AprevLayerM = multiplactiveEmulationFunctionPre(AprevLayerA)
					Walayer = Wa[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")]
					Wmlayer = Wm[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wm")]
					Za = tf.add(Za, tf.add(tf.matmul(AprevLayerA, Walayer), Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")]))	
					Zm = tf.add(Zm, tf.add(tf.matmul(AprevLayerM, Wmlayer), Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")]))
				Zm = multiplactiveEmulationFunctionPost(Zm)
			else:
				AprevLayerA = AprevLayer
				AprevLayerM = multiplactiveEmulationFunctionPre(AprevLayerA)
				Za = tf.add(tf.matmul(AprevLayerA, Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")]), Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")])
				Zm = tf.add(tf.matmul(AprevLayerM, Wm[generateParameterNameNetwork(networkIndex, l1, "Wm")]), Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")])
				Zm = multiplactiveEmulationFunctionPost(Zm)

			Aa = activationFunction(Za)
			Am = activationFunction(Zm)
			print("Za = ", Za)
			print("Zm = ", Zm)
			Z = tf.concat([Za, Zm], axis=1)
			A = tf.concat([Aa, Am], axis=1)
		
		if(debugOnlyTrainFinalLayer):
			if(l1 < numberOfLayers):
				A = tf.stop_gradient(A)

		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
						
		AprevLayer = A

	if(maxLayer == numberOfLayers):
		return tf.nn.softmax(Z)
	else:
		return A

def multiplactiveEmulationFunctionPre(AprevLayer):
	AprevLayerM = tf.math.log(AprevLayer)
	return AprevLayerM
	
def multiplactiveEmulationFunctionPost(ZmIntermediary):
	Zm = tf.math.exp(ZmIntermediary) 
	return Zm
	
def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	


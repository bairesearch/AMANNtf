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

specification: additiveMultiplicativeNet-31July2022a.png

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs

debugSmallBatchSize = False
debugOnlyTrainFinalLayer = False
debugSingleLayerNetwork = False
debugFastTrain = False

supportMultipleNetworks = False

supportSkipLayers = False

initialiseMultiplicativeUnitBiasNegative = False
if(initialiseMultiplicativeUnitBiasNegative):
	initialiseMultiplicativeUnitBiasNegativeOffset = -1.2

activationMaxVal = 10.0
multiplicativeEmulationFunctionOffsetVal = 1.0	#add/subtract
multiplicativeEmulationFunctionPreMinVal = 1e-9
multiplicativeEmulationFunctionPreMaxVal = 1e+9	#or activationMaxVal (effective)
multiplicativeEmulationFunctionPostMaxVal = 20.0

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
#number of additive or multiplicative units of actual hidden ie non-input/output layers = n_h[hiddenLayer]/2
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0

def defineTrainingParameters(dataset):
	global batchSize
	
	learningRate = 0.001
	if(debugSmallBatchSize):
		batchSize = 2
	else:
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
	
	#generateLargeNetwork=True, useEvenNumHiddenUnits=True to generate additional units for addition/multiplication
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=True, useEvenNumHiddenUnits=True)
	
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	print("numberOfLayers = ", numberOfLayers)
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
			
		for l1 in range(1, numberOfLayers+1):

			if(l1 == numberOfLayers):
				#CHECKTHIS: last layer (fully connected additive layer, no skip layers)
				n_hPreviousLayer = n_h[l1-1]
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
							n_hPreviousLayer = n_h[l2]
							n_hCurrentLayerA = calculateLayerNumAdditiveOrMultiplicativeUnits(l1)
							n_hCurrentLayerM = calculateLayerNumAdditiveOrMultiplicativeUnits(l1)
							Walayer = tf.Variable(randomNormal([n_hPreviousLayer, n_hCurrentLayerA]))
							Wmlayer = tf.Variable(randomNormal([n_hPreviousLayer, n_hCurrentLayerM]))
							Wa[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")] = Walayer
							Wm[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wm")] = Wmlayer
				else:	
					n_hPreviousLayer = n_h[l1-1]
					n_hCurrentLayerA = calculateLayerNumAdditiveOrMultiplicativeUnits(l1)
					n_hCurrentLayerM = calculateLayerNumAdditiveOrMultiplicativeUnits(l1)
					Walayer = tf.Variable(randomNormal([n_hPreviousLayer, n_hCurrentLayerA]))
					Wmlayer = tf.Variable(randomNormal([n_hPreviousLayer, n_hCurrentLayerM]))
					Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")] = Walayer
					Wm[generateParameterNameNetwork(networkIndex, l1, "Wm")] = Wmlayer
					
				n_hCurrentLayerA = calculateLayerNumAdditiveOrMultiplicativeUnits(l1)
				n_hCurrentLayerM = calculateLayerNumAdditiveOrMultiplicativeUnits(l1)
				Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")] = tf.Variable(tf.zeros(n_hCurrentLayerA))
				if(initialiseMultiplicativeUnitBiasNegative):
					Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")] = tf.Variable(tf.zeros(n_hCurrentLayerM)+initialiseMultiplicativeUnitBiasNegativeOffset)				
				else:
					Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")] = tf.Variable(tf.zeros(n_hCurrentLayerM))
							
			if(supportSkipLayers):
				n_hCurrentLayer = n_h[l1]
				Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_hCurrentLayer], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_hCurrentLayer], dtype=tf.dtypes.float32))

	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			n_hPreviousLayer = n_h[numberOfLayers-1]
			WlayerF = randomNormal([n_hPreviousLayer*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer = tf.Variable(WlayerF)
			BlayerF = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer= tf.Variable(BlayerF)	#not currently used

def calculateLayerNumAdditiveOrMultiplicativeUnits(l):
	n_hLayerAll = n_h[l]
	if((l > 0) and (l < numberOfLayers)):
		#number of units for actual hidden layers: concatenation of additive and multiplicative neurons (x2)
		n_hLayerAdditiveOrMultiplicative = n_hLayerAll//2
	else:
		n_hLayerAdditiveOrMultiplicative = n_hLayerAll
	return n_hLayerAdditiveOrMultiplicative
											
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
		#print("\nl1 = " + str(l1))		
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
					AprevLayerA = clipActivation(AprevLayerA)
					AprevLayerM = multiplicativeEmulationFunctionPre(AprevLayerA)
					Walayer = Wa[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wa")]
					Wmlayer = Wm[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wm")]
					Za = tf.add(tf.matmul(AprevLayerA, Walayer))
					Zm = tf.add(tf.matmul(AprevLayerM, Wmlayer))
				Zm = multiplicativeEmulationFunctionPost(Zm)
				Za = tf.add(Za, Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")])
				Zm = tf.add(Zm, Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")])
			else:
				AprevLayerA = AprevLayer
				AprevLayerA = clipActivation(AprevLayerA)
				AprevLayerM = multiplicativeEmulationFunctionPre(AprevLayerA)

				Walayer = Wa[generateParameterNameNetwork(networkIndex, l1, "Wa")]
				Wmlayer = Wm[generateParameterNameNetwork(networkIndex, l1, "Wm")]
				#print("Wmlayer = ", Wmlayer)
				Za = tf.matmul(AprevLayerA, Walayer)
				Zm = tf.matmul(AprevLayerM, Wmlayer)
				#print("Zm = ", Zm)
				Zm = multiplicativeEmulationFunctionPost(Zm)
				#print("Zm = ", Zm)
				Za = tf.add(Za, Ba[generateParameterNameNetwork(networkIndex, l1, "Ba")])
				Zm = tf.add(Zm, Bm[generateParameterNameNetwork(networkIndex, l1, "Bm")])
				
			Aa = activationFunction(Za)
			Am = activationFunction(Zm)
			Z = tf.concat([Za, Zm], axis=1)
			A = tf.concat([Aa, Am], axis=1)

			if(tf.reduce_any(tf.math.is_nan(A))):
				print("tf.reduce_any(tf.math.is_nan(A))")
				ex

		if(debugOnlyTrainFinalLayer):
			if(l1 < numberOfLayers):
				A = tf.stop_gradient(A)

		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
						
		AprevLayer = A
			
	if(maxLayer == numberOfLayers):
		pred = tf.nn.softmax(Z)
		return pred
	else:
		return A

def clipActivation(A):
	A = tf.clip_by_value(A, -activationMaxVal, activationMaxVal)
	return A
	
def multiplicativeEmulationFunctionPre(AprevLayer):
	AprevLayer = AprevLayer + multiplicativeEmulationFunctionOffsetVal
	AprevLayer = tf.clip_by_value(AprevLayer, multiplicativeEmulationFunctionPreMinVal, multiplicativeEmulationFunctionPreMaxVal)	
	AprevLayerM = tf.math.log(AprevLayer)
	return AprevLayerM
	
def multiplicativeEmulationFunctionPost(ZmIntermediary):
	ZmIntermediary = tf.clip_by_value(ZmIntermediary, -multiplicativeEmulationFunctionPostMaxVal, multiplicativeEmulationFunctionPostMaxVal)
	Zm = tf.math.exp(ZmIntermediary)
	Zm = Zm - multiplicativeEmulationFunctionOffsetVal
	return Zm
	
def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	


import tensorflow as tf
import numpy as np
import word2vec
import keras
import dataProcessor
import csv

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Embedding,Dense,Bidirectional,LSTM,Dropout,concatenate,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import Input,Model

class rnnRunner():

	wordIndex = {}
	maxlen = 60
	keyLen = 2
	keyList = []
	keyIndex = {}

	def get_index(self,sentence):#turn word to index
		sequence = []
		for word in sentence:
			try:
				sequence.append(self.wordIndex[word])
			except KeyError:
				pass
		return sequence

	def get_key_index(self,sentence):
		sequence = []
		for word in sentence:
			try:
				sequence.append(self.keyIndex[word])
			except KeyError:
				pass
		return sequence

	def getSeq(self):

		newVec = word2vec.word2vec()
		content = newVec.readData()

		sequence = list(map(self.get_index,content))

		sequence = pad_sequences(sequence, maxlen=self.maxlen,padding = 'post')

		return sequence

	def preProcessData(self,path = "data.csv"):
		processor = dataProcessor.dataProcessor()
		userID,keyword,content = processor.readTrainingData(path=path)
		content = processor.preprocessing(content)
		processor.outputResult(content)

	def processValidationData(self,path = "validation.csv"):
		processor = dataProcessor.dataProcessor()
		userID,keyword,content = processor.readTrainingData(path=path)
		keyContent = processor.preprocessing(content,mode = 1,keyList = self.keyList)
		keySeq = list(map(self.get_key_index,keyContent))
		keySeq = pad_sequences(keySeq,maxlen = 2,padding = 'post')
		return keySeq

	def getData(self):
		vec = word2vec.word2vec()
		label = vec.getLabelAndKey()
		trainingData,embedding_matrix,wordIndex,keyValue,key_embedding_matrix,keyIndex,keyList = vec.getTrainingData()

		self.keyList = keyList

		self.keyIndex = keyIndex
		
		self.wordIndex = wordIndex

		newLabeltrain = []

		self.maxlen = 60#the median of the reddit length is 58.5 so 60 is used
		trainingData = pad_sequences(trainingData, maxlen=self.maxlen,padding = 'post')#convert all the reddit to length of 65

		self.keyLen = 2
		keyValue = pad_sequences(keyValue, maxlen=self.keyLen,padding = 'post')
		
		#code to generate all 0 fake label to test the model
		newLabel = []
		for i in range(len(label)):
			newLabel.append(0)
		label = newLabel
		label = np.array(label)
		#end fake code here

		extentTrainingData = []
		for i in range(len(trainingData)):
			extentTrainingData.append([trainingData[i],keyValue[i]])

		label = keras.utils.to_categorical(label,num_classes = 5)#convert the label to one_hot number can be adjust to final category num

		data_train, data_test, label_train, label_test = train_test_split(#split the training set and testing set
			extentTrainingData,
			label,
			test_size=0.1,
			random_state=30)

		new_data_train = []
		new_key_train =	[]

		for i in range(len(data_train)):
			new_data_train.append(data_train[i][0])
			new_key_train.append(data_train[i][1])

		new_data_test = []
		new_key_test = []

		for i in range(len(data_test)):
			new_data_test.append(data_test[i][0])
			new_key_test.append(data_test[i][1])

		return new_data_train,new_key_train,new_data_test,new_key_test,label_train,label_test,embedding_matrix,key_embedding_matrix,self.maxlen,self.keyLen

class BLSTM():

	data_train = ""
	key_train = ""
	data_test = ""
	key_test = ""
	label_train = ""
	label_test = ""
	embedding_matrix = ""
	key_embedding_matrix = ""
	maxlen = ""

	def __init__(self,data_train,key_train,data_test,key_test,label_train,label_test,embedding_matrix,key_embedding_matrix,maxlen,keylen):
		self.data_train = data_train
		self.key_train = key_train
		self.data_test = data_test
		self.key_test = key_test
		self.label_train = label_train
		self.label_test = label_test
		self.embedding_matrix = embedding_matrix
		self.key_embedding_matrix = key_embedding_matrix
		self.maxlen = maxlen
		self.keylen = keylen

	def lstmModel(self,pretrained_weights = 'BLSTM.hdf5'):

		inputContent = Input(shape = (self.maxlen,),name = 'content')
		inputKey = Input(shape = (self.keylen,),name = 'key')

		content = Embedding(
				input_dim = self.embedding_matrix.shape[0],
				output_dim = self.embedding_matrix.shape[1],
				input_length = self.maxlen,
				weights = [self.embedding_matrix],#use the index to retrive the word vector from embedding_matrix
				trainable = False
				)(inputContent)

		key = Embedding(
				input_dim = self.key_embedding_matrix.shape[0],
				output_dim = self.key_embedding_matrix.shape[1],
				input_length = self.keylen,
				weights = [self.key_embedding_matrix],#use the index to retrive the word vector from embedding_matrix
				trainable = False
				)(inputKey)

		content = Bidirectional(LSTM(512,return_sequences=True,recurrent_dropout=0.1),merge_mode='ave')(content)
		content = Dropout(0.1)(content)
		content = Bidirectional(LSTM(512,return_sequences=True,recurrent_dropout=0.1),merge_mode='ave')(content)
		content = Dropout(0.1)(content)
		content = Bidirectional(LSTM(256,return_sequences=True,recurrent_dropout=0.1),merge_mode='ave')(content)
		content = Dropout(0.1)(content)
		
		content = LSTM(128,recurrent_dropout = 0.1)(content)
		content = Dropout(0.1)(content)

		
		key = Bidirectional(LSTM(10,recurrent_dropout = 0.1),merge_mode = 'ave')(key)
		key = Dropout(0.1)(key)

		x =  concatenate([content,key])


		x = Dense(128,activation = 'sigmoid')(x)

		x = Dropout(0.25)(x)

		output = Dense(5,activation = 'softmax')(x)

		model = Model(inputs=[inputContent,inputKey], outputs=[output])

		model.compile(
			optimizer=Adam(lr = 1e-4),
			loss="categorical_crossentropy",
			metrics=['accuracy']
		)
		
		if(pretrained_weights):
			model.load_weights(pretrained_weights)

		return model

	def train(self,model):

		model_checkpoint = ModelCheckpoint('BLSTM.hdf5',monitor = 'loss',verbose = 1,save_best_only = True)

		history = model.fit(
			x = [self.data_train,self.key_train],
			y = self.label_train,
			validation_data = [[self.data_test,self.key_test],self.label_test],
			batch_size = 10,
			epochs = 1,
			callbacks = [model_checkpoint]
		)

	def predictResult(self,model,rnnRunner,testDataPath = "validation.csv",outputPath = "class.csv"):

		rnnRunner.preProcessData(path = testDataPath)

		dataSeq = rnnRunner.getSeq()

		keySeq = rnnRunner.processValidationData(path = testDataPath)

		result = model.predict([dataSeq,keySeq],batch_size = 10)#predict the result

		i = 0

		resultList = []

		while i!=len(result):
			resultList.append(np.argmax(result[i]))#argmax to extract the max possibility of given 10 result of softmax layer
			i+=1

		with open(outputPath,"w",newline = "") as file:#write the result to file
			writer = csv.writer(file)
			writer.writerow(["TestLabel"])
			for result in resultList:
				writer.writerow([result])
			
			file.close()

def main():
	runner = rnnRunner()
	runner.preProcessData()
	data_train,key_train,data_test,key_test,label_train,label_test,embedding_matrix,key_embedding_matrix,maxlen,keylen = runner.getData()
	

	rnn = BLSTM(data_train,key_train,data_test,key_test,label_train,label_test,embedding_matrix,key_embedding_matrix,maxlen,keylen)
	model = rnn.lstmModel()
	rnn.train(model)
	rnn.predictResult(model,runner)	
	

if __name__ == '__main__':
	main()



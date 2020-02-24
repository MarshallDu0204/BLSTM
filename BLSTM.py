import tensorflow as tf
import numpy as np
import word2vec
import keras
import dataProcessor
import csv

from sklearn.utils import class_weight
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding,Dense,Bidirectional,LSTM,Dropout,TimeDistributed,Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model


class rnnRunner():

	wordIndex = {}
	maxlen = 60#100

	def get_index(self,sentence):#turn word to index
		sequence = []
		for word in sentence:
			try:
				sequence.append(self.wordIndex[word])
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


	def getData(self):
		vec = word2vec.word2vec()
		trainingData,embedding_matrix,wordIndex = vec.getTrainingData()
		label = vec.getLabel()

		self.wordIndex = wordIndex

		self.maxlen = 60#the median of the reddit length is 58.5 so 60 is used
		trainingData = pad_sequences(trainingData, maxlen=self.maxlen,padding = 'post')#convert all the reddit to length of 65

		label = keras.utils.to_categorical(label,num_classes = 6)#convert the label to one_hot number can be adjust to final category num

		data_train, data_test, label_train, label_test = train_test_split(#split the training set and testing set
			trainingData,
			label,
			test_size=0.1,
			random_state=42,
			shuffle=True)

		return data_train,data_test,label_train,label_test,embedding_matrix,self.maxlen

class BLSTM():

	data_train = ""
	data_test = ""
	label_train = ""
	label_test = ""
	embedding_matrix = ""
	maxlen = ""

	def __init__(self,data_train,data_test,label_train,label_test,embedding_matrix,maxlen):
		self.data_train = data_train
		self.data_test = data_test
		self.label_train = label_train
		self.label_test = label_test
		self.embedding_matrix = embedding_matrix
		self.maxlen = maxlen
		

	def lstmModel(self,pretrained_weights = None):

		model = Sequential()

		model.add(Embedding(
				input_dim = self.embedding_matrix.shape[0],
				output_dim = self.embedding_matrix.shape[1],
				input_length = self.maxlen,
				weights = [self.embedding_matrix],#use the index to retrive the word vector from embedding_matrix
				trainable = False
				))

		model.add(Bidirectional(LSTM(400,return_sequences=True,activation = 'tanh',recurrent_dropout=0.25),merge_mode='concat'))#merge two lstm together
		model.add(Dropout(0.5))

		model.add(TimeDistributed(Dense(200,activation='relu')))
		
		model.add(Flatten())

		model.add(Dense(100,activation = 'relu'))

		model.add(Dropout(0.5))
		
		model.add(Dense(6, activation='softmax'))#softmax output
		
		plot_model(model, to_file='model.png', show_shapes=True)

		model.compile(
			optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy']
		)

		if(pretrained_weights):
			model.load_weights(pretrained_weights)

		return model

	def train(self,model):

		model_checkpoint = ModelCheckpoint('Model.hdf5',monitor = 'loss',verbose = 1,save_best_only = True)

		history = model.fit(
			x = self.data_train,
			y = self.label_train,
			validation_data = [self.data_test,self.label_test],
			batch_size = 20,
			epochs = 30,
			callbacks = [model_checkpoint]
		)

	def predictResult(self,model,rnnRunner,testDataPath = "validation.csv",outputPath = "class.csv"):

		rnnRunner.preProcessData(path = testDataPath)

		dataSeq = rnnRunner.getSeq()

		result = model.predict(dataSeq,batch_size = 20)#predict the result

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
	data_train,data_test,label_train,label_test,embedding_matrix,maxlen = runner.getData()


	rnn = BLSTM(data_train,data_test,label_train,label_test,embedding_matrix,maxlen)
	model = rnn.lstmModel()
	rnn.train(model)
	rnn.predictResult(model,runner)
	

if __name__ == '__main__':
	main()

import tensorflow as tf
import numpy as np
import word2vec
import keras

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense,Bidirectional,LSTM,Dropout,TimeDistributed,Activation
from keras.optimizers import Adam

def getData():
	vec = word2vec.word2vec()
	trainingData,embedding_matrix = vec.getTrainingData()
	label = vec.getLabel()

	newLabeltrain = []

	maxlen = 60#the median of the reddit length is 58.5 so 60 is used
	trainingData = pad_sequences(trainingData, maxlen=maxlen,padding = 'post')#convert all the reddit to length of 65

	
	#code to generate all 0 fake label to test the model
	newLabel = []
	for i in range(len(label)):
		newLabel.append(0)
	label = newLabel
	label = np.array(label)
	#end fake code here

	label = keras.utils.to_categorical(label,num_classes = 5)#convert the label to one_hot number can be adjust to final category num

	data_train, data_test, label_train, label_test = train_test_split(#split the training set and testing set
		trainingData,
		label,
		test_size=0.1,
		random_state=30)

	return data_train,data_test,label_train,label_test,embedding_matrix,maxlen

class lstm():

	data_train = ""
	data_test = ""
	label_train = ""
	label_test = ""
	embedding_matrix = ""
	maxlen = ""
	model_type = 0

	def __init__(self,data_train,data_test,label_train,label_test,embedding_matrix,maxlen):
		self.data_train = data_train
		self.data_test = data_test
		self.label_train = label_train
		self.label_test = label_test
		self.embedding_matrix = embedding_matrix
		self.maxlen = maxlen

	def lstmModel(self):

		model = Sequential()

		model.add(Embedding(
				input_dim = self.embedding_matrix.shape[0],
				output_dim = self.embedding_matrix.shape[1],
				input_length = self.maxlen,
				weights = [self.embedding_matrix],#use the index to retrive the word vector from embedding_matrix
				trainable = False
				))

		model.add(Bidirectional(LSTM(128,return_sequences=True,recurrent_dropout=0.1),merge_mode='ave'))#merge two lstm together
		model.add(Dropout(0.1))
		model.add(Bidirectional(LSTM(128,return_sequences=True,recurrent_dropout=0.1),merge_mode='ave'))#merge two lstm together
		model.add(Dropout(0.1))
		model.add(LSTM(128,recurrent_dropout = 0.1))
		model.add(Dropout(0.1))
		model.add(Dense(128, activation='sigmoid'))
		model.add(Dropout(0.25))
		model.add(Dense(5, activation='softmax'))#softmax output
		

		model.compile(
			optimizer=Adam(lr = 1e-4),
			loss="categorical_crossentropy",
			metrics=['accuracy']
		)

		return model

	def train(self,model):
		history = model.fit(
			x = self.data_train,
			y = self.label_train,
			validation_data = [self.data_test,self.label_test],
			batch_size = 10,
			epochs = 5
		)

def runModel():
	data_train,data_test,label_train,label_test,embedding_matrix,maxlen = getData()
	rnn = lstm(data_train,data_test,label_train,label_test,embedding_matrix,maxlen)
	model = rnn.lstmModel()
	rnn.train(model)

runModel()

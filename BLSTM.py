import tensorflow as tf
import numpy as np
import word2vec
import keras
import dataProcessor
import csv

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding,Dense,Bidirectional,LSTM,Dropout,TimeDistributed,Flatten,InputSpec
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras import Input,Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints

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
		trainingData = pad_sequences(trainingData, maxlen=self.maxlen,padding = 'post')#convert all the reddit to length of 60

		label = keras.utils.to_categorical(label,num_classes = 2)#convert the label to one_hot number can be adjust to final category num

		data_train, data_test, label_train, label_test = train_test_split(#split the training set and testing set
			trainingData,
			label,
			test_size=0.2,
			random_state=42,
			shuffle=True)

		return data_train,data_test,label_train,label_test,embedding_matrix,self.maxlen

class Attention(Layer):

	def __init__(self, return_attention=False, **kwargs):
		self.init = initializers.get('uniform')
		self.supports_masking = True
		self.return_attention = return_attention
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
		assert isinstance(input_shape, list)

		self.w = self.add_weight(shape=(input_shape[1][2], 1),
								 name='{}_w'.format(self.name),
								 initializer=self.init)
		self._trainable_weights = [self.w]
		super(Attention, self).build(input_shape)

	def call(self, x):
		assert isinstance(x, list)

		s, h = x

		h_shape = K.shape(h)  
		d_w, T = h_shape[0], h_shape[1]
		logits = K.dot(h, self.w)
		logits = K.reshape(logits, (d_w, T))
		alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
		alpha = alpha / K.sum(alpha, axis=1, keepdims=True)

		r = K.sum(s * K.expand_dims(alpha), axis = 1)
		h_star = K.tanh(r)

		if self.return_attention:
			return [h_star, alpha]
		return h_star

	def compute_output_shape(self, input_shape):
		assert isinstance(input_shape, list)
		output_len = input_shape[1][2]
		if self.return_attention:
			return [(input_shape[1][0], output_len), (input_shape[1][0], input_shape[1][1])]
		return (input_shape[1][0], output_len)

	def compute_mask(self, input, input_mask=None):
		if isinstance(input_mask, list):
			return [None] * len(input_mask)
		else:
			return None

	def get_config(self):
		config = {'init':self.init,'supports_masking':self.supports_masking,'return_attention':self.return_attention}
		base_config = super(Attention,self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

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

		inputContent = Input(shape = (self.maxlen,),name = 'content')

		content = Embedding(
				input_dim = self.embedding_matrix.shape[0],
				output_dim = self.embedding_matrix.shape[1],
				input_length = self.maxlen,
				weights = [self.embedding_matrix],#use the index to retrive the word vector from embedding_matrix
				trainable = False
				)(inputContent)

		lstmContent = Bidirectional(LSTM(100,return_sequences=True,recurrent_dropout=0.25),merge_mode='ave')(content)
		lstmContent = Dropout(0.4)(lstmContent)

		lstmContent = Bidirectional(LSTM(100,return_sequences=True,recurrent_dropout=0.25),merge_mode='ave')(lstmContent)
		lstmContent = Dropout(0.4)(lstmContent)

		weightContent = Dense(200,activation = 'relu')(content)

		attContent = Attention()([lstmContent,weightContent])

		attContent = Dense(128,activation = 'relu')(attContent)

		attContent = Dropout(0.4)(attContent)

		output = Dense(2,activation = 'softmax')(attContent)

		model = Model(inputs=inputContent, outputs=[output])
		
		#plot_model(model, to_file='model.png', show_shapes=True)

		model.compile(
			optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['acc']
		)

		if(pretrained_weights):
			model.load_weights(pretrained_weights)

		return model

	def train(self,model):

		model_checkpoint = ModelCheckpoint('Model.hdf5',monitor = 'loss',verbose = 1,save_best_only = True)

		history = model.fit(
			x = self.data_train,
			y = self.label_train,
			validation_data = (self.data_test,self.label_test),
			batch_size = 20,
			epochs = 10,
			callbacks = [model_checkpoint]
		)

	def evaluate(self,model,testData,testLabel,modelPath = None):

		if(modelPath):
			model.load_weights(modelPath)

		result = model.predict(testData,batch_size = 20)#predict the result

		i = 0
		resultList = []
		while i!=len(result):
			resultList.append(np.argmax(result[i]))#argmax to extract the max possibility of given 10 result of softmax layer
			i+=1

		accNum = 0

		i = 0
		while i!=len(resultList):
			if(resultList[i] == np.argmax(testLabel[i])):
				accNum+=1
			i+=1
		accNum = accNum/len(resultList)

		print(accNum)



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

	rnn.evaluate(model,data_test,label_test)	
	#rnn.predictResult(model,runner)
	

if __name__ == '__main__':
	main()

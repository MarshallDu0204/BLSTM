from gensim.models import Word2Vec
import numpy as np
import csv

class word2vec():

	wordIndex = ""
	keyIndex = ""
	keyWord = []


	def getLabelAndKey(self,path = "data.csv"):#read the label
		label = []
		keyWord = []
		trainingData = []
		
		with open(path,'r',encoding = 'utf-8') as file:
			lines = csv.reader(file)
			for line in lines:
				trainingData.append(line)
			trainingData = trainingData[1:len(trainingData)-1]
			for element in trainingData:
				label.append(element[2])  
				keyWord.append(element[4].split("^"))
			self.keyWord = keyWord
		return label

	def readData(self):#read the processed data
		with open("result.txt","r",encoding = 'utf-8') as file:
			wordSet = set()
			lines = file.readlines()
			newContent = []
			for line in lines:
				line = line.split(" , ")
				newLine = []
				for word in line:
					word = word.strip()
					if word!='':
						newLine.append(word)
						wordSet.add(word)
				newContent.append(newLine)
		return newContent

	def model(self,newContent,size=400,window = 5,min_count = 2,mode = 0):

		model = Word2Vec(newContent,size = size,window = window, min_count=min_count, sg=1)#initialize the w2v model

		if mode == 0:

			wordList = list(model.wv.vocab.keys())#get the vocabulary list

			self.wordIndex = {word: index for index, word in enumerate(wordList)}#put index and vocabulary in dict

			return model,wordList

		if mode == 1:

			keyList = list(model.wv.vocab.keys())

			self.keyIndex = {word: index for index, word in enumerate(keyList)}

			return model,keyList

	def get_key_index(self,sentence):
		sequence = []
		for word in sentence:
			try:
				sequence.append(self.keyIndex[word])
			except KeyError:
				pass
		return sequence


	def get_index(self,sentence):#turn word to index
		sequence = []
		for word in sentence:
			try:
				sequence.append(self.wordIndex[word])
			except KeyError:
				pass
		return sequence

	def getTrainingData(self):

		newContent = self.readData()

		w2v_model,wordList = self.model(newContent)

		embeddings_matrix = np.zeros((len(wordList), w2v_model.vector_size))#init the embedding_matrix

		for i in range(len(wordList)):#add the corresponde wordvector with index into embedding metrix 
			embeddings_matrix[i] = w2v_model.wv[wordList[i]]
		
		trainingIndex = list(map(self.get_index,newContent))#get the sequenced sentence

		key_w2v_model,keyList = self.model(self.keyWord,size = 10,window = 1,min_count = 1,mode = 1)

		key_embedding_matrix = np.zeros((len(keyList), key_w2v_model.vector_size))

		for i in range(len(keyList)):
			key_embedding_matrix[i] = key_w2v_model.wv[keyList[i]]

		keyValue = list(map(self.get_key_index,self.keyWord))

		return trainingIndex,embeddings_matrix,self.wordIndex,keyValue,key_embedding_matrix,self.keyIndex,keyList



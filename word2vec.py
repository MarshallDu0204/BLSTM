from gensim.models import Word2Vec
import numpy as np
import csv

class word2vec():

	wordIndex = ""

	def getLabel(self,path = "data.csv"):#read the label
		label = []
		trainingData = []
		
		with open(path,'r',encoding = 'utf-8') as file:
			lines = csv.reader(file)
			for line in lines:
				trainingData.append(line)
			trainingData = trainingData[1:len(trainingData)-1]
			for element in trainingData:
				label.append(element[2])  

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

	def model(self,newContent):

		model = Word2Vec(newContent, size=128, window=5, min_count=2, sg=1)#initialize the w2v model

		wordList = list(model.wv.vocab.keys())#get the vocabulary list

		self.wordIndex = {word: index for index, word in enumerate(wordList)}#put index and vocabulary in dict

		return model,wordList

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
		return trainingIndex,embeddings_matrix



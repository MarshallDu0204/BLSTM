import numpy as np
import csv
import porter
import string
import matplotlib.pyplot as plt

class dataProcessor():

	stopSet = set()

	def __init__(self):
		self.stopSet = self.getStopSet()

	def readTrainingData(self,path = "data.csv"):
		trainingData = []
		userID = []
		keywords = []
		content = []
		with open(path,'r',encoding = 'utf-8') as file:
			lines = csv.reader(file)
			for line in lines:
				trainingData.append(line)
			trainingData = trainingData[1:len(trainingData)-1]
			for element in trainingData:
				userID.append(element[3])
				keywords.append(element[4])
				content.append(element[5])

		return userID,keywords,content

	def getStopSet(self,path = "stopwords.txt"):#construct the stopwords to set to minimize the time
		with open(path) as file:
			newStopWords = []
			stopwords = file.readlines()
			for element in stopwords:
				newStopWords.append(element.strip())
			stopwords = set(newStopWords)
			return stopwords

	def cWord(self,word):#remove \n at any position in the sentence
		emojiSet = {"🤢","🤔","😣","😀","😆","💖","🥰","🌻","💫","✨","🙏","🏼","😥","😉","👨‍💻","😊","😅","😌","😪","😎","❤️","😂","💜","💀","👍"}#remove all the emoji
		info = ''
		k = 0
		for element in word:
			if element!='\n' and element not in emojiSet:
				info = info+element
				k+=1
		return info,k


	def preprocessing(self,content):
		newContent = []
		contentLength = []
		p = porter.PorterStemmer()
		for line in content:
			newLine = []
			line = line.split(" ")
			for word in line:
				word = word.strip()#clean the '\n' 
				word = word.lower()	#change to lowercase
				word = p.stem(word)#stem the word
				cleanWord = word.translate(str.maketrans('','', string.punctuation)) #remove the punctuation
				if cleanWord not in self.stopSet:#remove stop word
					if len(cleanWord) != 0:#remove the empty element
						cleanWord,wordLen = self.cWord(cleanWord)
						if wordLen<25 and wordLen>=1:
							newLine.append(cleanWord)
			contentLength.append(len(newLine))
			
			newContent.append(newLine)
		# code to generate the histogram of the reddit length
		'''
		contentLength = np.array(contentLength)
		a = plt.hist(contentLength, bins='auto')
		plt.show()
		'''
		print(np.median(contentLength))
		return newContent

	def outputResult(self,content,contentpath = "result.txt"):
		with open(contentpath,"w",encoding = 'utf-8') as f:
			f.write("")
			f.close()
		with open(contentpath,"a",encoding = 'utf-8') as f:
			for line in content:
				info = ""
				for word in line:
					info = info+word+" , "
				info = info+"\n"
				f.write(info)
			f.close()

processor = dataProcessor()
userID,keyword,content = processor.readTrainingData()
content = processor.preprocessing(content)
processor.outputResult(content)
import csv

def readTrainingData(path = "data.csv"):
	
	result = []
	trainingData = []
	with open(path,'r',encoding = 'utf-8') as file:
		lines = csv.reader(file)
		for line in lines:
			trainingData.append(line)
		trainingData = trainingData[1:len(trainingData)-1]
		for element in trainingData:
			result.append(element[2])

	return result

def readResult(path = "class.csv"):
	result = readTrainingData()

	result1 = []
	trainingData = []
	with open(path,'r',encoding = 'utf-8') as file:
		lines = csv.reader(file)
		for line in lines:
			trainingData.append(line)
		trainingData = trainingData[1:len(trainingData)-1]
		for element in trainingData:
			result1.append(element[0])

	print(len(result1))
	print(len(result))
	k = 0
	for i in range(len(result1)):
		if result[i] == result1[i]:
			k+=1

	k = k/len(result1)

	print(k)

readResult()
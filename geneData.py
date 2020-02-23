import csv

def readTrainingData(path = "data.csv"):
		userA = []
		userB = []
		result = []
		userID = []
		keywords = []
		content = []
		trainingData = []
		with open(path,'r',encoding = 'utf-8') as file:
			lines = csv.reader(file)
			for line in lines:
				trainingData.append(line)
			trainingData = trainingData[1:len(trainingData)-1]
			for element in trainingData:
				userA.append(element[0])
				userB.append(element[1])
				result.append(element[2])
				userID.append(element[3])
				keywords.append(element[4])
				content.append(element[5])

		

		for i in range(len(content)):
			if result[i] == "":
				if userA[i]!="":
					result[i] = userA[i]
				else:
					result[i] = userB[i]

		for i in range(len(result)):
			if int(result[i])>5:
				print(result[i],i)
		
		return userA,userB,result,userID,keywords,content

def writeData(path = "data.csv"):
	userA,userB,result,userID,keywords,content = readTrainingData()
	with open(path,"w",newline = "",encoding = "utf-8") as file:#write the result to file
		writer = csv.writer(file)
		writer.writerow(["userA","userB","result","userID","keywords","content"])
		for i in range(len(userA)):
			writer.writerow([userA[i],userB[i],result[i],userID[i],keywords[i],content[i]])
		writer.writerow(['60 per group'])
		file.close()
		
readTrainingData()



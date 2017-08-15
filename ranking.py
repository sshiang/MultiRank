### paper ###

import os
import sys
import numpy as np
import math
import copy
import random
import copy
import operator


import argparse
import heapq
from collections import namedtuple

parser = argparse.ArgumentParser(description='MultiRank for object recognition improvement.')
parser.add_argument('-a', '--alpha', dest='alp', default='-1.0', type=float, help='alpha parameter in MultiRank.')
parser.add_argument('-n', '--number', dest='number', default='100', type=int, help='number of relations used.')
parser.add_argument('-g', '--gamma', dest='gamma', default=0.5, type=float, help='gamma')
parser.add_argument('-i', '--input-dir', dest='relationDir', default="../dataset/NYU_depth/relation_NYU/", help='Directory of relation input.')
parser.add_argument('-p', '--predict-dir', dest='predictDir', default="../dataset/NYU_depth/predict_NYU/", help='Directory of predict input.')
parser.add_argument('-o', '--output-dir', dest='outputDir', default='rerank_NYU/', help='Output directory.')
parser.add_argument('-l', '--label', dest='labelPath', default="../dataset/NYU_depth/labelList", help='label mapping file')
parser.add_argument('-m', '--method', dest='alg', default='cascade', help='algorithm of MultiRank. \n "cascade": using verbal spatial only')
parser.add_argument('-v', '--between', dest='between', default='spatial', help='between')
parser.add_argument('-w', '--within', dest='within', default='spatial', help='within')

parser.add_argument('-b', '--bounding-box', dest='bounding', default='oracle', help="The way to generate bounding box")
parser.add_argument('-d', '--dimension', dest='dimension', default=2, type=int, help="dimension of bounding box, 2 or 3")
parser.add_argument('-f', '--filter', dest='filter', default=False, type=bool, help="Use computer vision score theshold as update switch.")
opts = parser.parse_args()


class Box:
	def __init__(self,x1=-1,x2=-1,y1=-1,y2=-1,z1=-1,z2=-1):
		self.x1=x1
		self.x2=x2
		self.y1=y1
		self.y2=y2
		self.z1=z1
		self.z2=z2

class Layer:
	def __init__(self,answer, box, indexs, labels, scores, name, first, ratio):
		self.answer = answer
		self.box = box
		self.max_score = max(scores)#first
		self.indexs = indexs
		self.labels = labels
		self.scores = normalize_list(scores)
		self.rerank_scores = copy.deepcopy(normalize_list(scores))
		self.nodeNum = len(indexs)
		self.name = name
		self.max_score = max(scores)#first
		self.first = first
		self.ratio = ratio

def makeDirectory(path):
	if os.path.exists(path)==False:
		os.makedirs(path)

def readLabel(path):
	labelDictionary = {}
	for line in open(path):
		seg = line.strip().split("\t")
		labelDictionary[seg[0]]=seg[1]
	return labelDictionary

def readRelation(path, number):
	global labelDictionary
	global relationDictionry
	relationMap = {}

	rels = [line.strip() for line in open(path) if "behind" not in line and "infront" not in line]
	rex = [x for x in rels if "-1" not in x.split("\t") and "0" not in x.split("\t")]

	#random_index = randrange(0,len(rels))	
	if number >= len(rex):
		new_rels = rex
	else:
		new_rels= random.sample(rex,min(len(rex),number))
	
	for line in new_rels:

		seg = line.strip().split("\t")
		assert len(seg)==3, "relation file %s error"%(path)
		target = seg[1]
		source = seg[2]
		relation = seg[0]
		
		#assert target in labelDictionary, "wrong label: %s"%(target)
		#assert source in labelDictionary, "wrong label: %s"%(source)
		#assert relation in relationDictionary, "wrong relation: %s"%(relation)

		if relation not in relationMap:
			relationMap[relation]=[(target, source)] #   {target:source}
		else:
			relationMap[relation].append((target, source))
	return relationMap

def readConfusion(path, excludePath, labelPath):
	count = 0
	labels = {}
	for line in open(labelPath):
		label = line.strip().split("\t")[0]
		labels[label]=count
		count+=1

	#labels = [line.strip().split("\t")[0] for line in open(labelPath)]
	exclude = {name.replace(".txt",""):0 for name in os.listdir(excludePath)}
	confusion = [[0 for x in range(len(labels))] for y in range(len(labels))]
	for imagename in os.listdir(path):
		if imagename in exclude:
			continue
		#print imagename
		if imagename[0] == ".":
			continue
		for filename in os.listdir(os.path.join(path,imagename)):
			count = 0
			for line in open(os.path.join(path, imagename, filename)):
				line = line.strip()	
				count+=1
				if count == 1:
					answer = line	
					continue
				elif count == 2:
					continue
				elif count == 3:
					predict = line.split("\t")[0]
				else:
					break
			#confusion[labels[answer]][labels[predict]]+=1
			confusion[labels[predict]][labels[answer]]+=1

	
	for i in range(len(confusion)):
		total = sum(confusion[i])
		if total==0:
			continue
		for j in range(len(confusion[i])):
			confusion[i][j] = float(confusion[i][j])/float(total)
	
	return confusion

def readShutterstock(path, labelPath):
	labelMap = {}
	cc = 0
	for line in open(labelPath):
		labelMap[line.strip()]=cc
		cc +=1
	matrix = []
	for line in open(path):
		vector = line.strip().split(" ")
		vector = [int(x) for x in vector]
		matrix.append(vector)
	return normalize(matrix), labelMap

def readPrediction(path):
	count = 0
	predicts = {}
	labels = []
	scores = []
	indexs = {}
	#cc = 0
	name = path.split("/")[-1].replace(".txt","")
	for line in open(path):
		line = line.strip()
		count+=1
		if count == 1:
			if opts.bounding == "oracle":
				answer = {line:1.0}
			else:
				answer = {x.split(":")[0]:float(x.split(":")[1]) for x in line.split("\t")}
				#print answer	
			continue
		elif count == 2:
			if opts.dimension == 2:
				[x1,y1,x2,y2] = [int(x) for x in line.split(" ")]
				box = Box(x1,x2,y1,y2,-1,-1)
			else:
				[x1,x2,y1,y2,z1,z2] = [float(x) for x in line.split(" ")]
				box = Box(x1,x2,y1,y2,z1,z2)
			continue



		seg = line.split("\t")
		label = seg[0]
		score = float(seg[1])
		indexs[label] = len(labels)
		labels.append(label)
		scores.append(score)
		#scores = normalize_list(scores)
	
	correct = True
	first = scores[0]
	ratio = scores[0]/scores[1]
	first_score = scores[0]#/scores[1]
	if labels[0]!=answer:
		correct = False

	scores = normalize_list(scores)
	#print indexs

	return Layer(answer, box, indexs, labels, scores, name, first, ratio), correct, first_score

def converge(list1, list2):
	assert len(list1) == len(list2), "length mismatch"
	diff = sum([(list1[i]-list2[i])**2 for i in range(len(list1))])
	if diff < 10**(-6):
		return True
	else:
		return False

def detectRel(layers):
	global relationDictionary
	detect = {rel:[] for rel in relationDictionary}
	for i in range(len(layers)):
		for j in range(len(layers)):
			if i == j:
				continue
			ix1=layers[i].box.x1
			ix2=layers[i].box.x2
			iy1=layers[i].box.y1
			iy2=layers[i].box.y2
			iz1=layers[i].box.z1
			iz2=layers[i].box.z2

			jx1=layers[j].box.x1
			jx2=layers[j].box.x2
			jy1=layers[j].box.y1
			jy2=layers[j].box.y2
			jz1=layers[j].box.z1
			jz2=layers[j].box.z2


			if opts.dimension == 2:
				# left 
				if ix1<=jx1 and ix2<=jx2: # should overlap in y?
					if iy2>=jy1 or jy2>=iy1:
						detect["left"].append((i,j))
				# right
				if ix1>=jx1 and ix2>=jx2:
					if iy2>=jy1 or jy2>=iy1:
						detect["right"].append((i,j))
				# on
				if ix1>=jx1 and iy1>=jy1 and ix2<=jx2 and iy2<=jy2:
					detect["on"].append((i,j))
				# above
				if iy1<=jy1 and iy2<=jy2:
					if ix2>=jx1 or jx2>=ix1:
						detect["above"].append((i,j))
				# below
				if iy1>=jy1 and iy2>=jy2:
					if ix2>=jx1 or jx2>=ix1:
						detect["below"].append((i,j))
				# infront
				if (ix2>=jx1 or jx2>=ix1) and (iy2>=jy1 or jy2>=iy1):
					if iy1>=jy1 and iy2>=jy2:
						detect["infront"].append((i,j))

				if (iy2>=jy1 and iy1<=jy1) or (jy2>=iy1 and jy1<=iy1):
					if (ix2>=jx1 and ix1<=jx1) or (jx2>=ix1 and jx1<=ix1):
						detect["near"].append((i,j))
			else:


				# left 
				if iy1<=jy1 and iy2<=jy2: # should overlap in y?
					detect["above"].append((i,j))
				# right
				if iy1>=jy1 and iy2>=jy2:
					detect["below"].append((i,j))
				# on
				#if ix1>=jx1 and iy1>=jy1 and ix2<=jx2 and iy2<=jy2:
				if iy1>=jy1 and iy2<=jy2 and ix2<=jx2 and ix1>=jx1:
					detect["on"].append((i,j))
				# above
				if ix1<=jx1 and ix2<=jx2:
					detect["left"].append((i,j))
				# below
				if ix1>=jx1 and ix2>=jx2:
					detect["right"].append((i,j))
				# infront
				if iz1<=jz1 and iz2<=jz2:
					detect["infront"].append((i,j))
				# behind
				if iz1>=jz1 and iz2>=jz2:
					detect["behind"].append((i,j))
				# near
				#if near(box1,box2):
				#	detect["near"].append((i,j))	
				
	return detect

def addMap(relations,a,b,c,d):
	if a not in relations:
		relations[a]={b:[(c,d)]}
	elif b not in relations[a]:
		relations[a][b] = [(c,d)]
	else:
		relations[a][b].append((c,d))
	return relations

def detectRelation(layers, relationMap):
	relations = {}
	detects = detectRel(layers)
	for rel in relationMap:
		for nlp in relationMap[rel]:
			source = nlp[0]
			target = nlp[1]
			for pair in detects[rel]:
				sourceid = pair[0]
				targetid = pair[1]
				relations = addMap(relations, sourceid, source, targetid, target)
				relations = addMap(relations, targetid, target, sourceid, source)			
	return relations

def normalize(links):
	new_links = copy.deepcopy(links) 
	cc = 0
	for i in range(len(links)):
		ss = float(sum(links[i]))
		if ss == 0:
			cc += 1
			for j in range(len(links[i])):
				new_links[i][j] = 1.0/float(len(links[i]))	
		else:
			for j in range(len(links[i])):
				new_links[i][j] = links[i][j]/ss
	return new_links

def normalize_list(scores):
	total = sum(scores)
	scores = [scores[x]/float(total) for x in range(len(scores))]
	return scores

def isEmpty(matrix):
	for vector in matrix:
		if len(list(set(vector))) != 1:
			return False
	return True

def buildShutterWithin(layers):
	global occurance
	global labelMap

	links_shutter = []
	maxLabels = [layers[i].labels[layers[i].scores.index(max(layers[i].scores))] for i in range(len(layers))]
	for i in range(len(layers)):
		links = [[0 for x in range(layers[i].nodeNum)] for y in range(layers[i].nodeNum)]	
		totalScore = [0 for x in range(layers[i].nodeNum)]
		for j in range(layers[i].nodeNum):
			selfLabel = layers[i].labels[j]
			for k in range(len(layers)):
				if i == j:
					continue		
				#totalScore[j] += occurance[labelMap[selfLabel]][labelMap[maxLabels[k]]]
				totalScore[j] += occurance[labelMap[maxLabels[k]]][labelMap[selfLabel]]
		totalScore = normalize_list(totalScore)
		for j in range(len(links)):
			for k in range(len(links[j])):
				links[j][k] = totalScore[k]
		links_shutter.append(normalize(links))
	return links_shutter

def getConfusionWithin(layer):
	global labelMap
	global confusion
	matrix = [[0 for x in range(layer.nodeNum)] for y in range(layer.nodeNum)]
	for i in range(layer.nodeNum):
		for j in range(layer.nodeNum):
			matrix[i][j] = confusion[labelMap[layer.labels[i]]][labelMap[layer.labels[j]]]
	for i in range(layer.nodeNum):
		total = float(sum(matrix[i]))
		for j in range(layer.nodeNum):
			matrix[i][j] = matrix[i][j]/total
	return matrix


def getShutterBetween(layers, i, j):
	global occurance
	global labelMap
	matrix = [[0 for x in range(layers[i].nodeNum)] for y in range(layers[j].nodeNum)]
	for k in range(layers[i].nodeNum):
		for l in range(layers[j].nodeNum):
			matrix[l][k] = occurance[labelMap[layers[j].labels[l]]][labelMap[layers[i].labels[k]]]
	for k in range(layers[i].nodeNum):
		total = float(sum(matrix[i]))
		for l in range(layers[i].nodeNum):
			matrix[k][l] = matrix[k][l]/total
	
	return matrix

def buildRelationWithin(relations, layers):
	# build within link	
	links_within = []
	for i in range(len(layers)):
		links = [[0 for x in range(layers[i].nodeNum)] for y in range(layers[i].nodeNum)]
		if i in relations:
			for label in relations[i]:
				for j in range(layers[i].nodeNum):
					if label not in layers[i].indexs:
						continue
					links[j][layers[i].indexs[label]] += layers[i].scores[layers[i].indexs[label]]
		# then normalize
		links_within.append(normalize(links))
	return links_within

def getEqualMatrix(size):
	matrix = [[1.0/float(size) for x in range(size)] for y in range(size)]
	return matrix

def buildRelationBetween(relations, layers):
	# build between link

	equalMatrix = getEqualMatrix(len(layers))

	links_between = [[None for x in range(len(layers))] for y in range(len(layers))]
	for i in range(len(layers)):
		if i not in relations:
			for j in range(len(layers)):				
				links_between[i][j] = equalMatrix
			continue

		for j in range(len(layers)):
			links = [[0 for x in range(layers[j].nodeNum)] for y in range(layers[i].nodeNum)]
			for ilabel in relations[i]:
				for pair in relations[i][ilabel]:
					jindex = pair[0]
					jlabel = pair[1]
					if j == jindex:
						# match, j/all to i/ilabel and i/all to j/jlabel are strong
						if ilabel not in layers[i].indexs or jlabel not in layers[j].indexs:
							continue
						links[layers[i].indexs[ilabel]][layers[j].indexs[jlabel]]+=layers[j].scores[layers[j].indexs[jlabel]]

			links_between[i][j] = normalize(links)
	return links_between


def rerank(layers,relationMap, alpha, beta, alg):
	if opts.between == "spatial" or opts.within == "spatial":
		# detect match 
		relations = detectRelation(layers, relationMap)
	
	if opts.within == "spatial":
		#build within link
		links_within = buildRelationWithin(relations, layers)
	elif opts.within == "shutterstock":
		links_within = buildShutterWithin(layers)

	if opts.between == "spatial":
		links_between = buildRelationBetween(relations, layers)

	#if "shutterstock" in alg:
	#	links_shutter = buildShutterWithin(layers)


	# update until convergence
	iteration = 0
	while True:
		iteration +=1
		all_converge = True

		for i in range(len(layers)):
			rerank_scores = np.array([alpha*layers[i].scores[x] for x in range(layers[i].nodeNum)])
			notEmptyList = []
			
			beta = 0
			for j in range(len(layers)):
				# from j to i
				if isEmpty(links_between[j][i])==False:
					notEmptyList.append(j)
					beta += layers[j].max_score

			if beta == 0:
				#print "beta 0"
				continue

			# between
			temp_scores = [0 for x in range(layers[i].nodeNum)]
			for j in notEmptyList:
				if i != j:
					temp_scores += np.dot(layers[j].max_score/beta, np.dot(np.transpose(links_between[j][i]), layers[j].rerank_scores))
				else:
					temp_scores += np.dot((layers[j].max_score/beta),layers[j].rerank_scores)

			# within
			rerank_scores += (1-alpha)*np.dot(np.transpose(links_within[i]), temp_scores)


			'''
			for j in notEmptyList:
				if "cascade" in alg or "confusion" in alg or "shutterstock" in alg:
					if i == j:
						temp_vector = copy.deepcopy(layers[i].rerank_scores)
					else:
						temp_vector = [0 for x in range(layers[i].nodeNum)]

						matrix_between = getShutterBetween(layers, i, j)
						#print matrix_between
						for k in range(layers[i].nodeNum):
							for l in range(layers[j].nodeNum):
								
								if alg == "shutterstock" or alg == "cascade-all":
									global occurance
									global labelMap
									index_from = labelMap[layers[j].labels[l]]
									index_to = labelMap[layers[i].labels[k]]
									temp_vector[k] += matrix_between[l][k]*layers[j].rerank_scores[l]	

								else:
									temp_vector[k] += links_between[j][i][l][k]*layers[j].rerank_scores[l]
						if alg == "cascade-all":
							temp_vector3 = [0 for x in range(layers[i].nodeNum)]
							for k in range(layers[i].nodeNum):
								for l in range(layers[j].nodeNum):
									temp_vector3[k] += links_between[j][i][l][k]*temp_vector[l]
							temp_vector = copy.deepcopy(temp_vector3)

					# within j
					matrix = getConfusionWithin(layers[i])
					for k in range(layers[i].nodeNum):
						for l in range(layers[i].nodeNum):	
							if alg == "cascade":
								rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*links_within[i][l][k]*temp_vector[l]
								#rerank_scores[k] += (1-alpha)*(1.0/float(len(notEmptyList)))*links_within[i][l][k]*temp_vector[l]
							elif alg == "confusion":
								#index_from = labelMap[layers[j].labels[l]]
								#index_to = labelMap[layers[i].labels[k]]
								#rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*matrix[l][k]*temp_vector[l]
								rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*confusion[l][k]*temp_vector[l]
							elif alg == "shutterstock":
								#index_from = labelMap[layers[j].labels[l]]
								#index_to = labelMap[layers[i].labels[k]]
								#rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*links_shutter[i][l][k]*temp_vector[l]
								rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*matrix[l][k]*temp_vector[l]
								#rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*links_within[i][l][k]*temp_vector[l]

					if alg == "cascade-confusion" or alg == "cascade-all":
						temp_vector2 = [0 for x in range(layers[i].nodeNum)]
						for k in range(layers[i].nodeNum):
							for l in range(layers[i].nodeNum):
								#index_from = labelMap[layers[j].labels[l]]
								#index_to = labelMap[layers[i].labels[k]]
								#temp_vector2[k]+=matrix[l][k]*temp_vector[l]
								temp_vector2[k]+=confusion[l][k]*temp_vector[l]
						for k in range(layers[i].nodeNum):
							for l in range(layers[i].nodeNum):					
								rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*links_within[i][l][k]*temp_vector2[l]
					if alg == "separate-confusion":
						gama = float(sys.argv[4])
						for k in range(layers[i].nodeNum):
							for l in range(layers[i].nodeNum):	
								rerank_scores[k] += gama*layers[j].max_score/beta*links_within[i][l][k]*temp_vector[l]
								rerank_scores[k] += (1-alpha-gama)*layers[j].max_score/beta*confusion[l][k]*temp_vector[l]
				elif alg == "separate":
					if i == j:
						for k in range(layers[i].nodeNum):
							for l in range(layers[i].nodeNum):
								rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*links_within[i][l][k]*layers[i].rerank_scores[l]
					else:
						for k in range(layers[i].nodeNum):
							for l in range(layers[j].nodeNum):
								rerank_scores[k] += (1-alpha)*layers[j].max_score/beta*links_between[j][i][l][k]*layers[j].rerank_scores[l]
			'''
	
			if converge(layers[i].rerank_scores, rerank_scores) == False:
				all_converge = False

			layers[i].rerank_scores = copy.deepcopy(rerank_scores)
	
		if all_converge == True:
			break

	return layers

def scoreFilter(layers,first,ratio):
	for layer in layers:
		if layer.first >= first and layer.ratio >= ratio:
			layer.rerank_scores = copy.deepcopy(layer.scores)
	return layers


def countAccuracy(layer,sort,correct_original,correct_care_original,map_original,number_map,isCare,total):
	counting = 0.0
	counting_care = 0.0

	max_ans = max(layer.answer.iteritems(), key=operator.itemgetter(1))[0]
	if max_ans == "-1" or max_ans == "0":
		return correct_original, correct_care_original, map_original, number_map, total

	if layer.labels[sort[0]] in layer.answer:
		correct_original += 1#layer.answer[layer.labels[sort[0]]]
		if isCare:
			correct_care_original += 1
		#b+=layer.answer[layer.labels[sort[0]]]

	for j in range(len(sort)):
		if layer.labels[sort[j]] in layer.answer:
			counting += 1#layer.answer[layer.labels[sort[j]]]
			map_original += counting/float(j+1)
			number_map += 1

			#c += layer.answer[layer.labels[sort[j]]]/float(j+1)
			#index_original = j

	total += 1

	return correct_original, correct_care_original, map_original, number_map, total

def countForeAccuracy(layer,sort,correct_original,correct_care_original,map_original,number_map,isCare,total):
	counting = 0.0
	counting_care = 0.0


	if layer.labels[sort[0]] in layer.answer:
		correct_original += 1#layer.answer[layer.labels[sort[0]]]
		if isCare:
			correct_care_original += 1
		#b+=layer.answer[layer.labels[sort[0]]]

	for j in range(len(sort)):
		if layer.labels[sort[j]] in layer.answer:
			counting += 1#layer.answer[layer.labels[sort[j]]]
			map_original += counting/float(j+1)
			number_map += 1

			#c += layer.answer[layer.labels[sort[j]]]/float(j+1)
			#index_original = j

	total +=1

	return correct_original, correct_care_original, map_original, number_map, total



def checkCare(index,layer,relationMap,detects):
	for rel in detects:
		if rel not in relationMap:
			continue
		for pair in detects[rel]:
			if index == pair[0]:
				for answerPair in relationMap[rel]:
					if answerPair[0] in layer.answer:
						return True
			elif index == pair[1]:
				for answerPair in relationMap[rel]:
					if answerPair[1] in layer.answer:
						return True

	return False


def precision_and_recall(layers, sort, relationMap, detects):
	global labelDictionary
	careObjects = {}
	#for obj in labelDictionary:
	#	careObjects[obj] = 1

	for rel in relationMap:
		for pair in relationMap[rel]:
			#if pair[0] in careObjects:
			#	del careObjects[pair[0]]
			#if pair[1] in careObjects:
			#	del careObjects[pair[1]]
			careObjects[pair[0]] = 1
			careObjects[pair[1]] = 1

	precision = 0
	precision_all = 0
	recall = 0
	recall_all = 0
	for i in range(len(layers)):
		for obj in careObjects:
			if obj in layers[i].answer:
				if layers[i].labels[sort[0]] in layers[i].answer:
					precision +=1
				precision_all += 1

			if layers[i].labels[sort[0]] == obj:
				if layers[i].labels[sort[0]] in layers[i].answer:
					recall += 1
				recall_all += 1
	
	if precision_all == 0:
		p = 0
	else:
		p = float(precision)/float(precision_all)

	if recall == 0:
		r = 0
	else:
		r = float(recall)/float(recall_all)
				
		# if predict in answer and predict 
	return p,r

rels = ["left", "right", "above", "below", "on", "infront", "behind", "near"]
relationDictionary = {x:0 for x in rels}

labelPath = opts.labelPath # "../dataset/NYU_depth/labelList"
labelDictionary = readLabel(labelPath)

'''
relationDir = sys.argv[1]
predictDir = sys.argv[2]
outputDir = sys.argv[3]
'''
#relationDir = "/Users/eternal0815/research/dataset/NYU_depth/nyudv2-wm/autoRelation_NYU"
#relationDir = "../dataset/NYU_depth/auto_relation_NYU"
#relationDir = "../dataset/NYU_depth/relation_NYU"
#relationDir = "../dataset/NYU_depth/correct_relation_NYU_2"

#relationDir = "../dataset/NYU_depth/auto_relation_error_NYU/"
#relationDir = "../dataset/NYU_depth/valid_relation_NYU/"
#predictDir = "../dataset/NYU_depth/predict_NYU"
#outputDir = "rerank_NYU"


filenames = os.listdir(opts.relationDir)

#confusion = readConfusion(opts.predictDir, opts.relationDir, "../dataset/NYU_depth/label.txt")

#occurance, labelMap = readShutterstock("../dataset/occurance-matrix.txt", "../dataset/NYU_depth/label.txt")

#occurance, labelMap = readShutterstock("matrix_all.txt", "../dataset/NYU_depth/label.txt")


ACC = []
ACC_R = []
ACC_F = []

MRR = []
MRR_R = []
MRR_F = []



#alphas = [0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]
#alphas = [opts.alp] 
if opts.alp < 0:
	alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
else:
	alphas = [opts.alp]
#alphas = [0.25,0.5,0.75]
#alphas = [0.6]
#alphas = [float(sys.argv[3])]



differentLevelMRR = []
differentLevelACC = []
differentLevelMRR_R = []
differentLevelACC_R = []

select = {x:0 for x in range(40)}
for i in range(11,16):
	select[i] = 1
for i in range(16,21):
	select[i] = 2
for i in range(21,26):
	select[i] = 3
for i in range(26,41):
	select[i] = 4



for alpha in alphas:

	ps = []	
	rs = []
	ps_rerank = []
	rs_rerank = []


	correct_original = 0
	correct_rerank = 0
	correct_filter = 0
	correct_care_original = 0
	correct_care_rerank = 0
	correct_care_filter = 0
	map_original_all = 0
	map_rerank_all = 0
	map_filter_all = 0


	total = 0
	total_original, total_rerank, total_filter = 0,0,0
	total_care = 0
 
	filenames = os.listdir(opts.relationDir)

	right = []
	wrong = []


	single_ACC = []
	single_MRR = []
	single_ACC_rerank = []
	single_MRR_rerank = []

	single_select_ACC = []
	single_select_MRR = []
	single_select_ACC_rerank = []
	single_select_MRR_rerank = []


	for imagename in os.listdir(opts.relationDir):
		print imagename

		if imagename[0] == ".": # or "_1.txt" in imagename:
			continue


		layers = [] 
		imageid = imagename.replace(".txt","")

		relationMap = readRelation("%s/%s"%(opts.relationDir,imagename), opts.number)


		if os.path.exists("%s/%s"%(opts.predictDir,imageid)) == False:
			continue

		for filename in os.listdir("%s/%s"%(opts.predictDir,imageid)):
			if filename[0] == ".":
				continue
			layer, correctness, first_score = readPrediction("%s/%s/%s"%(opts.predictDir, imageid, filename))
			layers.append(layer)

			if correctness == True:
				right.append(first_score)
			else:
				wrong.append(first_score)

		beta = sum([x.max_score for x in layers])

		# random walk
		layers = rerank(layers, relationMap, alpha, beta, opts.alg)

		if opts.filter==True:
			layers = scoreFilter(layers,0.6,10)

		# dump
		makeDirectory("%s/%s"%(opts.outputDir,imageid))	
		for layer in layers:
			f = open("%s/%s/%s.txt"%(opts.outputDir,imageid,layer.name),"w")
			f.write("%s\n"%("\t".join(["%s:%f"%(x,layer.answer[x]) for x in layer.answer])))
			f.write("%d %d %d %d\n"%(layer.box.x1,layer.box.y1,layer.box.x2,layer.box.y2))
			for i in range(layer.nodeNum):
				f.write("%s\t%s\n"%(layer.labels[i],layer.rerank_scores[i])) 
			f.close()

		# evaluate
		#total += len(layers)
		#if isCare:
		#	total+=len(layers)
		a = 0
		b = 0
		c = 0.0
		d = 0.0

		# should be refactored afterward:
		detects = detectRel(layers)

		for i in range(len(layers)):
			layer = layers[i]
			
			isCare = checkCare(i,layer,relationMap,detects)#checkCare(layer,relationMap)
			if isCare:	
				total_care += 1
			#else:
			#	print isCare

			map_original = 0
			map_rerank = 0
			map_filter = 0

			number_map = 0
			number_rerank_map = 0
			number_filter_map = 0

			sort_rerank = sorted(range(len(layer.rerank_scores)), key=lambda k: layer.rerank_scores[k], reverse=True)
			sort = sorted(range(len(layer.scores)), key=lambda k: layer.scores[k], reverse=True)

			'''
			if layer.labels[sort_rerank[0]] in layer.answer:
				correct_rerank += 1#layer.answer[layer.labels[sort_rerank[0]]]
				a+=layer.answer[layer.labels[sort_rerank[0]]]
			if layer.labels[sort[0]] in layer.answer:
				correct_original += 1#layer.answer[layer.labels[sort[0]]]
				b+=layer.answer[layer.labels[sort[0]]]
			index_original = 20
			index_rerank = 20
			'''

			correct_original,correct_care_original,map_original,number_map, total_original = countAccuracy(layer,sort,correct_original,correct_care_original,map_original,number_map,isCare, total_original)
			correct_rerank,correct_care_rerank,map_rerank,number_rerank_map, total_rerank = countAccuracy(layer,sort_rerank,correct_rerank,correct_care_rerank,map_rerank,number_rerank_map,isCare, total_rerank)


			'''
			counting = 0.0
			for j in range(len(sort)):
				if layer.labels[sort[j]] in layer.answer:
					counting += 1#layer.answer[layer.labels[sort[j]]]
					map_original += counting/float(j+1)
					number_map += 1
					c += layer.answer[layer.labels[sort[j]]]/float(j+1)
					index_original = j

			counting = 0.0
			for j in range(len(sort_rerank)):
				if layer.labels[sort_rerank[j]] in layer.answer:
					counting += 1#layer.answer[layer.labels[sort_rerank[j]]]
					map_rerank += counting/float(j+1)
					number_rerank_map += 1
					d += layer.answer[layer.labels[sort_rerank[j]]]/float(j+1)
					index_rerank = j
				#break
			'''

			if number_rerank_map!=0:		
				map_rerank_all += float(map_rerank)/float(number_rerank_map)
			if number_map != 0:
				map_original_all += float(map_original)/float(number_map)

		'''
		p,r = precision_and_recall(layers, sort, relationMap, detects)		
		ps.append(p)
		rs.append(r)

		p,r = precision_and_recall(layers, sort_rerank, relationMap, detects)		
		ps_rerank.append(p)
		rs_rerank.append(r)
		'''

		for layer in layers:
			if layer.first >= 0.6 and layer.ratio >= 10:
				layer.rerank_scores = copy.deepcopy(layer.scores)

		
		for i in range(len(layers)):
			layer = layers[i]
			sort_filter = sorted(range(len(layer.rerank_scores)), key=lambda k: layer.rerank_scores[k], reverse=True)

			correct_filter,correct_care_filter,map_filter,number_filter_map, total_filter = countAccuracy(layer,sort_filter,correct_filter,correct_care_filter,map_filter,number_filter_map,isCare,total_filter)


			'''
			if layer.labels[sort_filter[0]] in layer.answer:
				correct_filter += 1#layer.answer[layer.labels[sort_filter[0]]]

			countint = 0 
			for j in range(len(sort_filter)):
				if layer.labels[sort_filter[j]] in layer.answer:
					counting += 1#layer.answer[layer.labels[sort_filter[j]]]
					map_filter += counting/float(j+1)

			if index_rerank > index_original:
				print "x\t",imageid,"\t" , index_original,"\t", index_rerank ,"\t",layer.name
			elif index_rerank < index_original:	
				print "o\t",imageid,"\t" , index_original,"\t", index_rerank ,"\t",layer.name
			'''

		'''
		single_ACC.append(float(b)/float(len(layers)))
		single_ACC_rerank.append(float(a)/float(len(layers)))
		single_MRR.append(float(c)/float(len(layers)))
		single_MRR_rerank.append(float(d)/float(len(layers)))
		'''

	'''
	ACC.append(str(float(correct_original)/float(total)))
	ACC_R.append(str(float(correct_rerank)/float(total)))
	ACC_F.append(str(float(correct_filter)/float(total)))
	MRR.append(str(float(map_original_all)/float(total)))
	MRR_R.append(str(float(map_rerank_all)/float(total)))
	MRR_F.append(str(float(map_filter_all)/float(total)))
	'''	


	'''
	print "right ", np.mean(right), np.std(right)
	print "wrong ", np.mean(wrong), np.std(wrong)

	'''
	print "accuracy: %f"%(float(correct_original)/float(total_original))
	print "rerank accuracy: %f"%(float(correct_rerank)/float(total_rerank))

	print "care accuracy: %f"%(float(correct_care_original)/float(total_care))
	print "rerank care accuracy: %f"%(float(correct_care_rerank)/float(total_care))

	print "map: %f"%(float(map_original_all)/float(total_original))
	print "rerank map: %f"%(float(map_rerank_all)/float(total_rerank))

	'''	
	print np.mean(ps), np.mean(rs)
	print np.mean(ps_rerank), np.mean(rs_rerank)
	'''


	'''
	print single_ACC
	print single_ACC_rerank
	print single_MRR
	print single_MRR_rerank
	'''

'''
print " ".join(ACC)
print " ".join(ACC_R)
print " ".join(ACC_F)
print " ".join(MRR)
print " ".join(MRR_R)
print " ".join(MRR_F)
'''

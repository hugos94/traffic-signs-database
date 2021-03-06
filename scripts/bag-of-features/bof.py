import os, cv2, numpy, copy, time

from matplotlib import pyplot

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from sklearn import svm

from sklearn.metrics import accuracy_score,classification_report,cohen_kappa_score,confusion_matrix

def get_images_paths(folder_path):
	images_paths = []
	quantity = []
	classes = []
	class_id = 0
	datasets_sum = 0
	
	subfolders_paths = os.listdir(folder_path)

	for image_class in subfolders_paths:
		folder_name = os.path.join(folder_path,image_class)
		class_path = [os.path.join(folder_name, f) for f in os.listdir(folder_name)]
		images_paths += class_path
		classes += [class_id]*len(class_path)
		class_id += 1
		quantity.append(len(class_path))
		datasets_sum += len(class_path)
	
	#Uncomment the code below to see info about dataset distribuition
	'''
	pyplot.bar( numpy.arange( class_id ), quantity, align='center' )
	pyplot.xlabel('Class')
	pyplot.ylabel('Number of training examples')
	pyplot.xlim([-1, class_id])
	pyplot.show()
	'''

	#Uncomment the code below to see quantity about dataset classes
	'''
	for index, classe in enumerate(subfolders_paths):
		print("{} - {} - {}".format(index, classe, quantity[index]))
	'''
	return images_paths

def detect_and_compute_keypoints(image_paths):
	# Detect and compute keypoints found on images
	detector = cv2.xfeatures2d.SIFT_create()
	descriptor = cv2.xfeatures2d.SIFT_create()
	data = []
	target = []
	for image_path in image_paths:
		image = cv2.imread(image_path)
		keypoints = detector.detect(image, None)
		(_ , description) = descriptor.compute(image, keypoints)
		image_path = image_path.split('\\')[-2]
		data.append(description)
		target.append(image_path)
	return (data, target)

def stack_descriptors(X_train):
	data = copy.deepcopy(X_train)
	descriptors = data.pop(0)
	for descriptor in data:
		descriptors = numpy.concatenate((descriptors, descriptor), axis=0)
	return descriptors

def main(dataset_path, clustering_method, n_clusters, classifier):
	print("Getting images paths\n")
	images_paths = get_images_paths(dataset_path)
	
	start = time.time()
	print("Detecting and computing keypoints")
	(data, target) = detect_and_compute_keypoints(images_paths)
	end = time.time()
	print("Time: {} seconds or {} minutes\n".format(end-start, (end-start)/60))

	print("Performing K Fold Separation\n")
	skf = StratifiedKFold(n_splits = 5, shuffle=False)
	skf.get_n_splits(data,target)

	iteration = 0
	accuracy_list, kappa_list = [], []
	for train_index, test_index in skf.split(data, target):
		classifier_copy = copy.deepcopy(classifier)
		clustering_method_copy = copy.deepcopy(clustering_method)
		print("Iteração: {}".format(iteration))
		startIteration = time.time()

		X_train, y_train, X_test, y_test = [], [], [], []

		for index in train_index:
			X_train.append(data[index])
			y_train.append(target[index])

		for index in test_index:
			X_test.append(data[index])
			y_test.append(target[index])

		print("Stacking descriptors")
		descriptors = stack_descriptors(X_train)

		print("Performing clustering")
		start = time.time()
		clustering_method_copy.fit(descriptors)
		end = time.time()
		print("Time spent on clustering: {} seconds or {} minutes\n".format(end-start, (end-start)/60))

		print("Creating codebook from traning fold")
		im_features = numpy.zeros((len(X_train), n_clusters), "float32")
		i = 0
		for descriptor in X_train:
			words = clustering_method_copy.predict(descriptor)
			for w in words:
				im_features[i][w] += 1
			i += 1

		print("Standardize training features")
		stdSlr = StandardScaler().fit(im_features)
		im_features = stdSlr.transform(im_features)

		print("Training Classifier")
		start = time.time()
		classifier_copy.fit(im_features, numpy.array(y_train))
		end = time.time()
		print("Time spent to training classifier: {} seconds or {} minutes\n".format(end-start, (end-start)/60))

		print("Creating codebook from testing fold")
		im_features_test = numpy.zeros((len(X_test), n_clusters), "float32")
		i = 0
		for descriptor in X_test:
			words = clustering_method_copy.predict(descriptor)
			for w in words:
				im_features_test[i][w] += 1
			i += 1

		print("Standardize testing features")
		stdSlr = StandardScaler().fit(im_features_test)
		im_features_test = stdSlr.transform(im_features_test)

		print("Making predictions")
		start = time.time()
		predictions = [prediction for prediction in classifier_copy.predict(im_features_test)]
		end = time.time()
		print("Time spent on prediction: {} seconds or {} minutes\n".format(end-start, (end-start)/60))

		print("Calculating metrics")
		accuracy = accuracy_score(y_test,predictions)
		cohen_kappa = cohen_kappa_score(y_test,predictions)
		accuracy_list.append(accuracy)
		kappa_list.append(cohen_kappa)
		report = classification_report(y_test,predictions)

		cnf_matrix = confusion_matrix(y_test,predictions)

		print("*****Acuracy: {}*****".format(accuracy))
		print("*****Kappa:   {}*****".format(cohen_kappa))
		#print(report)
		#print("Matriz de confusão:\n{}\n\n".format(cnf_matrix))

		endIteration = time.time()
		print("Total time of iteration {}: {} seconds or {} minutes\n".format(iteration, endIteration-startIteration, (endIteration-startIteration)/60))

		iteration += 1

	accuracy_list = numpy.array(accuracy_list)
	print("#####Mean Acuracy: %0.4f (+/- %0.4f)#####" % (accuracy_list.mean(), accuracy_list.std() * 2))
	kappa_list = numpy.array(kappa_list)
	print("#####Mean Kappa:   %0.4f (+/- %0.4f)#####" % (kappa_list.mean(), kappa_list.std() * 2))

if __name__ == '__main__':
	start = time.time()
	dataset_path = ".\..\..\..\\17keslon_augmentation_flip+rotation+projection_resize_120"
	n_clusters = 500
	clustering_method = MiniBatchKMeans(n_clusters=n_clusters)
	#classifier = svm.LinearSVC() #One-vs-All
	classifier = svm.SVC() #One-vs-One
	main(dataset_path, clustering_method, n_clusters, classifier)
	end = time.time()
	print("\nTotal time: {} seconds or {} minutes\n".format(end-start, (end-start)/60))


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
	detector = cv2.xfeatures2d.SURF_create()
	descriptor = cv2.xfeatures2d.SURF_create()
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

def main():
	print("Getting images paths")
	dataset_path = ".\..\..\datasets\initial-dataset"
	images_paths = get_images_paths(dataset_path)
	
	start = time.time()
	print("Detecting and computing keypoints")
	(data, target) = detect_and_compute_keypoints(images_paths)
	end = time.time()
	print("Time: {} seconds or {} minutes\n".format(end-start, (end-start)/60))

	print("Performing K Fold Separation")
	skf = StratifiedKFold(n_splits = 5, shuffle=False)
	skf.get_n_splits(data,target)

if __name__ == '__main__':
	start = time.time()
	main()
	end = time.time()
	print("\nTotal time: {} seconds or {} minutes\n".format(end-start, (end-start)/60))


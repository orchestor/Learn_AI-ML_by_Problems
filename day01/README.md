# Problem:
Training a model which can recognize the handwritten digits from a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

tips:
1. This is multi-classification problem
2. methods = [svm, K-nearest neighbors]


# solution 1: SVM without preprocessing the images
You will get an accuracy of 10%. That's bad.

# solution 2: SVM with transforming color images into black and white ones.
The accuracy is improved to 88.7%. Wow.

test_images[test_images>0]=1
train_images[train_images>0]=1

# solution 3: SVM with Penalty parameter, rbf kernal, gamma
svm.SVC(kernel = 'rbf', C = 7, gamma = 0.009)
The accuracy is improved to 93.8%
# Reference
1. http://yann.lecun.com/exdb/mnist/
2. https://www.kaggle.com/c/digit-recognizer
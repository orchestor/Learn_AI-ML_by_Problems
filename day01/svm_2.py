import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm


# Prepare the input training and test sets

labeled_images = pd.read_csv('./input/train.csv')
images = labeled_images.iloc[0:5000, 1:]
labels = labeled_images.iloc[0:5000, :1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
test_images[test_images>0]=1
train_images[train_images>0]=1

# Observe the sample of dataset
i = 1
img = train_images.iloc[i].as_matrix()
img = img.reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i,0])
plt.show()
plt.hist(train_images.iloc[i])
plt.show()


# Train the model
clf = svm.SVC(kernel= 'rbf', C = 7, gamma = 0.009)
clf.fit(train_images, train_labels.values.ravel())

# Evaluate the model
score = clf.score(test_images,test_labels)
print("The current accuracy is ", score)

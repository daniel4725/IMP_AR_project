import cv2
import numpy as np
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


def getFiles(train, path):
    images = []
    count = 0
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    if (train is True):
        np.random.shuffle(images)

    return images


def readImage(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (150, 150))


def getDescriptors(sift, img):
    # kp, des = orb.detectAndCompute(img, None)
    # kp, des = brisk.detectAndCompute(img, None)
    kp, des = sift.detectAndCompute(img, None)  # kp = keypoints des = descriptors
    return des


def vstackDescriptors(descriptor_list):  # append verticaly the descriptors.
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    return descriptors


def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    return kmeans


def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])  # matrix of no_clusters cols and image_count rows
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, -1)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features  # matrix of the number of the k features on each image


def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3, 0.01, 1, 0.005, 0.003, 0.001, 5, 10]
    gammas = [0.1, 0.11, 0.095, 0.105, 0.01, 1, 0.2, 0.7, 0.4, 0.05, 0.07, 0.03]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def findSVM(im_features, train_labels, kernel):
    features = im_features
    if kernel == "precomputed":
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)

    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param)
    # svm = SVC(kernel=kernel, C=4, tol=10**(-5), class_weight='balanced')
    svm.fit(features, train_labels)
    return svm


def trainModel(path, no_clusters, kernel):
    images = getFiles(True, path)
    print("Train images path detected.")
    sift = cv2.SIFT_create()
    # orb = cv2.ORB_create()
    # brisk = cv2.BRISK_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 4
    image_count = len(images)

    for img_path in images:
        if("menu" in img_path):
            class_index = 0
        elif("triangle" in img_path):
            class_index = 1
        elif("circle" in img_path):
            class_index = 2
        else:
            class_index = 3

        train_labels = np.append(train_labels, class_index)
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        # des = getDescriptors(orb, img)
        # des = getDescriptors(brisk, img)
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)
    print("Train images normalized.")

    # plotHistogram(im_features, no_clusters)
    # print("Features histogram plotted.")

    svm = findSVM(im_features, train_labels, kernel)
    print("SVM fitted.")
    print("Training completed.")

    file = open('bovw.pkl', 'wb')

    pickle.dump([kmeans, scale, svm, im_features, train_labels, no_clusters], file)

    file.close()

    return kmeans, scale, svm, im_features


def plotConfusionMatrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plotConfusions(true, predictions):
    np.set_printoptions(precision=2)

    class_names = ["menu", "triangle", "circle", "square"]
    plotConfusionMatrix(true, predictions, classes=class_names,
                      title='Confusion matrix, without normalization')

    plotConfusionMatrix(true, predictions, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    plt.show()


def findAccuracy(true, predictions):
    print('accuracy score: %0.3f' % accuracy_score(true, predictions))


def testModel(path, kmeans, scale, svm, im_features, no_clusters, kernel):
    test_images = getFiles(False, path)
    print("Test images path detected.")

    count = 0
    true = []
    descriptor_list = []

    name_dict = {
        "0": "menu",
        "1": "triangle",
        "2": "circle",
        "3": "square",
    }

    sift = cv2.SIFT_create()
    # orb = cv2.ORB_create()
    # brisk = cv2.BRISK_create()


    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)
        # des = getDescriptors(orb, img)
        # des = getDescriptors(brisk, img)

        if (des is not None):
            count += 1
            descriptor_list.append(des)

            if ("menu" in img_path):
                true.append("menu")
            elif ("triangle" in img_path):
                true.append("triangle")
            elif ("circle" in img_path):
                true.append("circle")
            else:
                true.append("square")

    descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)

    test_features = scale.transform(test_features)

    kernel_test = test_features
    if (kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)

    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print("Test images classified.")

    plotConfusions(true, predictions)
    print("Confusion matrixes plotted.")

    findAccuracy(true, predictions)
    print("Accuracy calculated.")
    print("Execution done.")


def execute(train_path, test_path, no_clusters, kernel):
    kmeans, scale, svm, im_features = trainModel(train_path, no_clusters, kernel)
    testModel(test_path, kmeans, scale, svm, im_features, no_clusters, kernel)


if __name__ == '__main__':

    execute(train_path='Train', test_path='Test', no_clusters=200, kernel='linear')


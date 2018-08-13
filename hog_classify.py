import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import pickle

from extractfeatures import ExtractFeature

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

class HogClassifier:

    def __init__(self):
        self.sample_size = 500
        self.colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.extractfeatures = ExtractFeature()


    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extract_features(self, imgs, cspace='RGB', orient=9,
                         pix_per_cell=8, cell_per_block=2, hog_channel=0,spatial_size=(32, 32),
                         hist_bins=32, spatial_feature=True, hist_feature=True, hog_feature=True ):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
        for file in imgs:
            img_features=[]
            # Read in each one by one
            img =cv2.imread(file)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else:
                feature_image = np.copy(image)

            if spatial_feature == True:
                spatial_features = self.extractfeatures.bin_spatial(feature_image, size=spatial_size)
                img_features.append(spatial_features)
            if hist_feature == True:
                # Apply color_hist()
                hist_features = self.extractfeatures.color_hist(feature_image, nbins=hist_bins)
                img_features.append(hist_features)
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_feature == True:
                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_image.shape[2]):
                        hog_features.append(self.extractfeatures.get_hog_features(feature_image[:, :, channel],
                                                             orient, pix_per_cell, cell_per_block,
                                                             vis=False, feature_vec=True))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self.extractfeatures.get_hog_features(feature_image[:, :, hog_channel], orient,
                                                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
                # Append the new feature vector to the features list
                img_features.append(hog_features)
            features.append(np.concatenate(img_features))
        # Return list of feature vectors
        return features


    def visualizehogfeatures(self, imgfile, cspace='RGB', car_image=True):

        image = cv2.imread(imgfile)
        carimage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        convcarimage = np.copy(carimage)
        if cspace != 'RGB':
            if cspace == 'HSV':
                convcarimage = cv2.cvtColor(carimage, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                convcarimage = cv2.cvtColor(carimage, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                convcarimage = cv2.cvtColor(carimage, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                convcarimage = cv2.cvtColor(carimage, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                convcarimage = cv2.cvtColor(carimage, cv2.COLOR_RGB2YCrCb)
        else:
            convcarimage = np.copy(carimage)

        #for channel in range(convcarimage.shape[2]):
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(convcarimage.shape[2]):
                hog_features.append(self.extractfeatures.get_hog_features(convcarimage[:, :, channel],
                                                                          self.orient, self.pix_per_cell, self.cell_per_block,
                                                                          vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features, hogimage = self.extractfeatures.get_hog_features(convcarimage[:, :, self.hog_channel], self.orient,
                                                                 self.pix_per_cell, self.cell_per_block, vis=True,
                                                                 feature_vec=True)

        title=''
        if car_image:
            title = 'Car Image'
        else:
            title = 'Not Car Image'

        #cv2.imwrite(output_dir + str(channel)+ '_' + os.path.basename(imgfile), hogimage)
        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(carimage, cmap='gray')
        plt.title(title+self.colorspace)
        plt.subplot(122)
        plt.imshow(hogimage, cmap='gray')
        plt.title('HOG Visualization')
        plt.show()
        return

    def classify_images(self, cars, notcars, savemodel=True):

        # Reduce the sample size because HOG features are slow to compute
        # The quiz evaluator times out after 13s of CPU time

        # cars = cars[0:self.sample_size]
        # notcars = notcars[0:self.sample_size]

        ### TODO: Tweak these parameters and see how the results change.

        t = time.time()
        spatialsize = (32, 32)
        histbins = 32
        car_features = self.extract_features(cars, cspace=self.colorspace, orient=self.orient,
                                        pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_size=(32, 32),
                                        hist_bins=32)
        notcar_features = self.extract_features(notcars,cspace=self.colorspace,orient=self.orient,
                                        pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_size=(32, 32),
                                        hist_bins=32)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract HOG features...')

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.1, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block', self.colorspace, 'color space',
              self.hog_channel, 'hog channel')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t = time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these', n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')
        if savemodel:
            # get attributes of our svc object
            # svc = dist_pickle["svc"]
            # X_scaler = dist_pickle["scaler"]
            # orient = dist_pickle["orient"]
            # pix_per_cell = dist_pickle["pix_per_cell"]
            # cell_per_block = dist_pickle["cell_per_block"]
            # spatial_size = dist_pickle["spatial_size"]
            # hist_bins = dist_pickle["hist_bins"]
            dist_pickle = {}
            dist_pickle["svc"] = svc
            dist_pickle["scaler"] = X_scaler
            dist_pickle["orient"] = self.orient
            dist_pickle["pix_per_cell"] = self.pix_per_cell
            dist_pickle["cell_per_block"] = self.cell_per_block
            dist_pickle["spatial_size"] = spatialsize
            dist_pickle["hist_bins"] = histbins
            pickle.dump(dist_pickle, open('svc_pickle.p', 'wb'))
        return

if __name__ == "__main__":
    hogimg_output_dir_name = './output_images/hog_features/'

    car_images = glob.glob('./examples/trainingset/vehicles/**/*.PNG', recursive=True)
    not_car_images = glob.glob('./examples/trainingset/non-vehicles/**/*.PNG', recursive=True)

    hogclassifier = HogClassifier()
    # ind = np.random.randint(0, len(car_images))
    # colorspace = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
    # for color in colorspace:
    #     hogclassifier.colorspace = color
    #     hogclassifier.hog_channel = 2
    #     hogclassifier.visualizehogfeatures(car_images[ind], True)
    #     hogclassifier.visualizehogfeatures(not_car_images[ind], True)

    # hogchannel = [0, 1, 2, 'ALL']
    # for color in colorspace:
    #     for hog in hogchannel:
    #         hogclassifier.colorspace = color
    #         hogclassifier.hog_channel = hog
    #         hogclassifier.classify_images(car_images, not_car_images)

    #Train and save 'ycrcb' hog classifier
    hogclassifier.colorspace = 'YCrCb'
    hogclassifier.hog_channel = 'ALL'
    hogclassifier.classify_images(car_images, not_car_images,True)

    print('function main called')
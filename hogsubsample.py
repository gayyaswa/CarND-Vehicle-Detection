import matplotlib.pyplot as plt
import numpy as np

import pickle
import cv2

from extractfeatures import ExtractFeature
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from collections import deque


class HogSubSampler:
    def __init__(self):
        self.extractfeature = ExtractFeature()
        # load a pe-trained svc model from a serialized (pickle) file
        self.dist_pickle = pickle.load(open("svc_pickle.p", "rb"))

        # get attributes of our svc object
        self.svc = self.dist_pickle["svc"]
        self.X_scaler = self.dist_pickle["scaler"]
        self.orient = self.dist_pickle["orient"]
        self.pix_per_cell = self.dist_pickle["pix_per_cell"]
        self.cell_per_block = self.dist_pickle["cell_per_block"]
        self.spatial_size = self.dist_pickle["spatial_size"]
        self.hist_bins = self.dist_pickle["hist_bins"]
        self.ystart = 400
        self.ystop = 656
        self.scale = 1.5
        self.heatmaps = deque(maxlen=10)

        return

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale):
        draw_img = np.copy(img)
        #img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = self.extractfeature.convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient * self.cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = self.extractfeature.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = self.extractfeature.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = self.extractfeature.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

        bbox = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.pix_per_cell
                ytop = ypos * self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features =self.extractfeature.bin_spatial(subimg, size=self.spatial_size)
                hist_features = self.extractfeature.color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)
                # test_features = self.X_scaler.transform(hog_features.reshape(1, -1))
                # test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    bbox.append(((xbox_left, ytop_draw + ystart),
                    (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

        return draw_img, bbox

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img

    def sumheatmapoverframes(self, heatmap, threshold):
        self.heatmaps.append(heatmap)
        heatmapsum = np.sum(self.heatmaps, axis=0)
        return self.apply_threshold(heatmapsum, threshold)

    def processimage(self, image, visualization=False):

        hog_out_img, bbox_list = hogsubsampler.find_cars(image, self.ystart, self.ystop, self.scale)

        if visualization:
            plt.imshow(hog_out_img)
            plt.show()

        heat = np.zeros_like(image[:, :, 0]).astype(np.float)

        # Add heat to each box in box list
        heat = self.add_heat(heat, bbox_list)

        # Apply threshold to help remove false positives
        heat = self.sumheatmapoverframes(heat,6)
        #heat = self.apply_threshold(heat, 2)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = hogsubsampler.draw_labeled_bboxes(np.copy(image), labels)

        if visualization:
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(draw_img)
            plt.title('Car Positions')
            plt.subplot(122)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')
            fig.tight_layout()
            plt.show()

        return  draw_img


if __name__ == "__main__":
    # img =cv2.imread('./test_images/test6.jpg')
    # image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hogsubsampler = HogSubSampler()
    hogsubsampler.scale = 1.5
    # draw_img = hogsubsampler.processimage(image, True)

    test_output = "test.mp4"
    clip = VideoFileClip("project_video.mp4")
    test_clip = clip.fl_image(hogsubsampler.processimage)
    test_clip.write_videofile(test_output, audio=False)
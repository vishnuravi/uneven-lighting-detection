import numpy as np
import cv2, sys
from scipy.signal import find_peaks
import json
import os

class Detector():
    def process_image(self, image): 
        #image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.5,
            minNeighbors=5,
            minSize=(200, 200)
        )

        filename = "cropped_face.jpg"

        results = {}

        if len(faces) > 0:
            x = faces[0][0]
            y = faces[0][1]
            w = faces[0][2]
            h = faces[0][3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            print("[INFO] Object found. Saving locally.")
            cv2.imwrite(filename, roi_color)
        else:
            results['error'] = 'No faces detected'
            return json.dumps(results)

        image = cv2.imread(filename)

        os.remove(filename)

        # image category
        category = 0

        # gray convertion
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height = image.shape[0]
        width = image.shape[1]

        # First test. Does the image have any big white spots?
        saturation_threshold = 255
        raw_saturation_region = cv2.threshold(image_gray, saturation_threshold, 255,  cv2.THRESH_BINARY)[1]
        num_raw_saturation_regions, raw_saturation_regions,stats, _ = cv2.connectedComponentsWithStats(raw_saturation_region)

        # index 0 is the background -> to remove
        area_raw_saturation_regions = stats[1:,4]

        min_area_bad_spot = 1000 # this can be calculated as percentage of the image area
        if (area_raw_saturation_regions.size > 0):
            if (np.max(area_raw_saturation_regions) > min_area_bad_spot):
                category = 2 # there is at least one spot
                results['acceptable'] = False
                results['issue'] = 'bright spot'
                print("unacceptable bc bright spots")

        # Second test. Is the image dark?   
        min_mean_intensity = 60

        if category == 0 :    
            mean_intensity = np.mean(image_gray)

            if (mean_intensity < min_mean_intensity):
                category = 3 # dark image
                results['acceptable'] = False
                results['issue'] = 'dark'
                print("unacceptable bc dark")

        window_len = 15 # odd number
        delay = int((window_len-1)/2)  # delay is the shift introduced from the smoothing. It's half window_len

        # for example if the window_len is 15, the delay is 7
        # infact hist.shape = 256 and smooted_hist.shape = 270 (= 256 + 2*delay)

        if category == 0:  
            perceived_brightness = self.get_perceived_brightness(image)
            hist,bins = np.histogram(perceived_brightness.ravel(),256,[0,256])

            # smoothed_hist is shifted from the original one    
            smoothed_hist = self.smooth(hist,window_len)
            
            # smoothed histogram syncronized with the original histogram
            sync_smoothed_hist = smoothed_hist[delay:-delay]    
            
            # if number the peaks with:
            #    20<bin<250
            #    prominance >= mean histogram value
            # the image could have shadows (but it could have also a background with some colors)
            mean_hist = int(height*width / 256)

            peaks, _ = find_peaks(sync_smoothed_hist, prominence=mean_hist)
            
            selected_peaks = peaks[(peaks > 20) & (peaks < 250)]
            print(selected_peaks)
            
            if (selected_peaks.size>1):
                category = 4 # there are shadows
                results['acceptable'] = False
                results['issue'] = 'shadows'
                print("unacceptable bc shadows")

        # all tests are passed. The image is ok
        if (category == 0):
            results['acceptable'] = True
            results['issue'] = 'none'
            print("acceptable")
        
        json_data = json.dumps(results)
        return json_data

    def get_perceived_brightness(self, float_img):
        float_img = np.float64(float_img)  # unit8 will make overflow
        b, g, r = cv2.split(float_img)
        float_brightness = np.sqrt((0.241 * (r ** 2)) + (0.691 * (g ** 2)) + (0.068 * (b ** 2)))
        brightness_channel = np.uint8(np.absolute(float_brightness))
        return brightness_channel
        
# from: https://stackoverflow.com/questions/46300577/find-locale-minimum-in-histogram-1d-array-python
    def smooth(self, x, window_len=11,window='hanning'):
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len<3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y

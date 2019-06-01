# coding: utf-8

#all template match code adapted from the DeepMoon project
#https://github.com/silburt/DeepMoon/blob/master/utils/template_match_target.py

import numpy as np
from skimage.feature import match_template
import cv2
import os
import sys
from keras.models import Model, load_model
#from keras.layers import Dense, Input, BatchNormalization, Dropout
#from keras.layers.merge import Add
#from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D
#from keras import backend as K
#from keras import optimizers
#from keras import metrics
import datetime
import glob
import random
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

class Match_Tiles(object):
    '''
    Match_Tiles is a test harness for running a variety of matching modifications quickly
    __init__: 
      version is looking for a string slug for the model, to be used in file names
      data_path is the string that will be passed to the glob function to grab files
      targ_path is the string that will be passed to the glob function to grab files
      VERBOSE is set to True by default, used for debugging, also used to decide whether images
        are printed to the screen
    '''
    
    #Recommended import: from match_utils import Match_Tiles as mt
    
        
    """
    Tuned Crater Detection Hyperparameters
    --------------------------------------
    minrad, maxrad : ints
        radius range in match_template to search over.
    longlat_thresh2, rad_thresh : floats
        if ((x1-x2)^2 + (y1-y2)^2) / min(r1,r2)^2 < longlat_thresh2 and
        abs(r1-r2) / min(r1,r2) < rad_thresh, remove (x2,y2,r2) circle (it is
        a duplicate of another crater candidate). In addition, when matching
        CNN-detected rings to corresponding csvs (i.e. template_match_t2c),
        the same criteria is used to determine a match.
    template_thresh : float
        0-1 range. If match_template probability > template_thresh, count as 
        detection.
    target_thresh : float
        0-1 range. target[target > target_thresh] = 1, otherwise 0
    rw : int
        1-32 range. Ring width, thickness of the rings used to match craters.
    """

    longlat_thresh2_ = 1.8
    rad_thresh_ = 1.0
    template_thresh_ = 0.5
    minrad_ = 6
    maxrad_ = 140
    target_thresh_ = 0.1
    rw_ = 8
        
    def __init__(self, model_version, model_path, data_path, targ_path, csv_path, rw=8, minrpx=7, maxrpx=140, tt=0.4,
                 RANDOMIZED=True, VERBOSE=True, log_str=''):        

        #Defaults tuned by DeepMoon team, all have '_' after
        self.longlat_thresh2_ = 1.8
        self.rad_thresh_ = 1.0
        self.template_thresh_ = 0.5
        self.minrad_ = 6
        self.maxrad_ = 140
        self.target_thresh_ = 0.1
        self.rw_ = 8

        #string name of model version, used for saving
        self.version = model_version
        
        self.verbose = VERBOSE
        #load files from paths
        self.data_arr = grab_files_list(data_path, self.verbose)
        self.targ_arr = grab_files_list(targ_path, self.verbose)
        self.csv_hu_arr = grab_files_list(csv_path, self.verbose)
        
        #load model
        self.model = load_model(model_path)
        
        #crater coord params
        self.coords_arr = None
        self.rw = rw #8 or 4
        self.minr_px = minrpx #6 #2km = 8.6 px
        self.maxr_px = maxrpx #140 #32 km = 138.2 px
        self.targ_thresh = tt
        
        #set up logging capabilities
        #docs: https://docs.python.org/2.3/lib/node304.html
        logger_name = 'test_log' + get_time()
        self.logger = logging.getLogger(logger_name)
        hdlr = logging.FileHandler('log/match_test_log_'+str(model_version)+'_'+get_time()+'.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr) 
        self.logger.setLevel(logging.INFO)
        self.logger.info('New log file created for this test: '+ model_version)
        #report the model_path, data_path, targ_path, & csv_path
        self.logger.info('Model path: '+ model_path)
        self.logger.info('Data path: '+ data_path)
        self.logger.info('Target path: '+ targ_path)
        self.logger.info('CSV path: '+ csv_path)
        
#    def run_match_all_tiles(self):
#        #load files from paths
#        #loop over number of tiles
#        #check they are valid 
#        return None
    
    
    def run_match_one_tile(self, data_fn, targ_fn, csv_px_fn): #move this out of self-land        
        data = []
        target = []

        loadIm(data_fn, targ_fn, data, target, step=512, newpx = 512, px = 7680)

        data = 2*np.array(data)-1
        target = np.array(target)

        print(data.shape)

        # Load model
        #fn_model = 'models/model_v13d_epochs_500_20180510_0458.h5'# d_epochs_500_20180508_0343.h5' #selected arbitrarily, v13a didn't work?
        #mod = load_model(fn_model)
        mod = self.model
        print('Model loaded at: ' + get_time())
        self.logger.info('Model loaded at: ' + get_time())

        # Run model on one tile's worth of data
        outs = mod.predict(data)
        print('Prediction finished at: ' + get_time())
        self.logger.info('Prediction finished at: ' + get_time())
            
        #Make the model output back into a tile
        tile_pred = remake_tile(outs, tile_size=7680, SHOW=False)
        
        #Make the orig data & target back into a tile (this should match the input target)
        tile_data = remake_tile(data, tile_size=7680, SHOW=False)
        tile_targ = remake_tile(target, tile_size=7680, SHOW=False)
        print('Tiles put back together at: ' + get_time())
        self.logger.info('Tiles put back together at: ' + get_time())
        
        #make copy of tile_pred *because the template match changes the np array directly
        copy_tile_pred = np.copy(tile_pred)
        
        #call crater_match
        tile_crater_coords = self.template_match_t(copy_tile_pred, minrad=self.minr_px, maxrad=self.maxr_px,
                 longlat_thresh2=self.longlat_thresh2_, rad_thresh=self.rad_thresh_,
                 template_thresh=self.template_thresh_,
                 target_thresh=self.targ_thresh, rw=self.rw)
        print('Coordinates determined from prediction at: ' + get_time())
        self.logger.info('Coordinates determined from prediction at: ' + get_time())
        
        #make image showing comparison 
        #crater_list_to_image(crater_array, img_size=2048)
        tile_found = crater_list_to_image(tile_crater_coords, img_size=7680)
        print('Crater list in new image finished at: ' + get_time())
        self.logger.info('Crater list in new image finished at: ' + get_time())
        
        #four_image(data_image, targ_image, pred_image, find_image, start_x=0, start_y=0, wid_ht=1024)
        four_image(tile_data, tile_targ, tile_pred, tile_found, start_x=0, start_y=0, wid_ht=1024)
        
        return tile_pred, tile_crater_coords 
    
    def run_compare_one_tile(self, csv_px_fn, tile_pred, list_coords=None):      
        csv_px_xyr = make_csv_px_array(csv_px_fn)
        csv_coords = np.copy(csv_px_xyr)
        
        copy_tile_pred = np.copy(tile_pred)
        #\
        stats, err, frac_dupes, templ_coords = self.template_match_t2c(copy_tile_pred, 
                    csv_coords, templ_coords=list_coords, 
                    minrad=self.minr_px, maxrad=self.maxr_px, 
                    longlat_thresh2=self.longlat_thresh2_, rad_thresh=self.rad_thresh_, 
                    template_thresh=self.template_thresh_, target_thresh=self.targ_thresh, 
                    rw=self.rw, rmv_oor_csvs=0)
        
        N_match, N_csv, N_detect, maxr = stats #maybe add frac_dupes to stats?
        err_lo, err_la, err_r = err
        
        #""""""
        #    Returns
        #    -------
        #    N_match : int
        #        Number of crater matches between your target and csv.
        #    N_csv : int
        #        Number of csv entries
        #    N_detect : int
        #        Total number of detected craters from target.
        #    maxr : int
        #        Radius of largest crater extracted from target.
        #    err_lo : float
        #        Mean longitude error between detected craters and csvs.
        #    err_la : float
        #        Mean latitude error between detected craters and csvs.
        #    err_r : float
        #        Mean radius error between detected craters and csvs.
        #    frac_dupes : float
        #""""""

        print('Number of matches: ' + str(N_match))
        print('Number of csv entries: ' + str(N_csv))
        print('Number of detected craters: ' + str(N_detect))
        print('Max radius: ' + str(maxr))
        print('err_lo: ' + str(err_lo))
        print('err_la: ' + str(err_la))
        print('err_r: ' + str(err_r))
        print('frac_dupes: ' + str(frac_dupes))
        
        return stats, err, frac_dupes, templ_coords, csv_px_xyr
    
    def template_match_t(self, target, minrad=minrad_, maxrad=maxrad_,
                     longlat_thresh2=longlat_thresh2_, rad_thresh=rad_thresh_,
                     template_thresh=template_thresh_,
                     target_thresh=target_thresh_, rw=rw_):
        """Extracts crater coordinates (in pixels) from a CNN-predicted target by
        iteratively sliding rings through the image via match_template from
        scikit-image.
        Parameters
        ----------
        target : array
            CNN-predicted target.
        minrad : integer
            Minimum ring radius to search target over.
        maxrad : integer
            Maximum ring radius to search target over.
        longlat_thresh2 : float
            Minimum squared longitude/latitude difference between craters to be
            considered distinct detections.
        rad_thresh : float
            Minimum fractional radius difference between craters to be considered
            distinct detections.
        template_thresh : float
            Minimum match_template correlation coefficient to count as a detected
            crater.
        target_thresh : float
            Value between 0-1. All pixels > target_thresh are set to 1, and
            otherwise set to 0.
        Returns
        -------
        coords : array
            Pixel coordinates of successfully detected craters in predicted target.
        """

        # thickness of rings for template match
        #commented out because this is passed now
        #rw = 8 #default 2 from DeepMoon project, we use 8 or 4

        # threshold target
        target[target >= target_thresh] = 1
        target[target < target_thresh] = 0

        radii = np.arange(minrad, maxrad + 1, 1, dtype=int)
        coords = []     # coordinates extracted from template matching
        corr = []       # correlation coefficient for coordinates set
        for r in radii:
            # template
            n = 2 * (r + rw + 1)
            template = np.zeros((n, n))
            cv2.circle(template, (r + rw + 1, r + rw + 1), r, 1, rw)

            # template match - result is nxn array of probabilities
            result = match_template(target, template, pad_input=True)
            index_r = np.where(result > template_thresh)
            coords_r = np.asarray(list(zip(*index_r)))
            corr_r = np.asarray(result[index_r])

            # store x,y,r
            if len(coords_r) > 0:
                for c in coords_r:
                    coords.append([c[1], c[0], r])
                for l in corr_r:
                    corr.append(np.abs(l))

        # remove duplicates from template matching at neighboring radii/locations
        coords, corr = np.asarray(coords), np.asarray(corr)
        i, N = 0, len(coords)
        while i < N:
            Long, Lat, Rad = coords.T
            lo, la, r = coords[i]
            minr = np.minimum(r, Rad)

            dL = ((Long - lo)**2 + (Lat - la)**2) / minr**2
            dR = abs(Rad - r) / minr
            index = (dR < rad_thresh) & (dL < longlat_thresh2)
            if len(np.where(index == True)[0]) > 1:
                # replace current coord with max match probability coord in
                # duplicate list
                coords_i = coords[np.where(index == True)]
                corr_i = corr[np.where(index == True)]
                coords[i] = coords_i[corr_i == np.max(corr_i)][0]
                index[i] = False
                coords = coords[np.where(index == False)]
            N, i = len(coords), i + 1

        return coords

    
    def template_match_t2c(self, target, csv_coords, templ_coords=None, minrad=minrad_, maxrad=maxrad_,
                           longlat_thresh2=longlat_thresh2_,
                           rad_thresh=rad_thresh_, template_thresh=template_thresh_,
                           target_thresh=target_thresh_, rw=rw_, rmv_oor_csvs=0):
        """Extracts crater coordinates (in pixels) from a CNN-predicted target and
        compares the resulting detections to the corresponding human-counted crater
        data.
        Parameters
        ----------
        target : array
            CNN-predicted target.
        csv_coords : array
            Human-counted crater coordinates (in pixel units).
        minrad : integer
            Minimum ring radius to search target over.
        maxrad : integer
            Maximum ring radius to search target over.
        longlat_thresh2 : float
            Minimum squared longitude/latitude difference between craters to be
            considered distinct detections.
        rad_thresh : float
            Minimum fractional radius difference between craters to be considered
            distinct detections.
        template_thresh : float
            Minimum match_template correlation coefficient to count as a detected
            crater.
        target_thresh : float
            Value between 0-1. All pixels > target_thresh are set to 1, and
            otherwise set to 0.
        rmv_oor_csvs : boolean, flag
            If set to 1, remove craters from the csv that are outside your
            detectable range.
        Returns
        -------
        N_match : int
            Number of crater matches between your target and csv.
        N_csv : int
            Number of csv entries
        N_detect : int
            Total number of detected craters from target.
        maxr : int
            Radius of largest crater extracted from target.
        err_lo : float
            Mean longitude error between detected craters and csvs.
        err_la : float
            Mean latitude error between detected craters and csvs.
        err_r : float
            Mean radius error between detected craters and csvs.
        frac_dupes : float
            Fraction of craters with multiple csv matches.
        """
        # get coordinates from template matching IF they are not passed
        if(templ_coords is None):
            templ_coords = template_match_t(target, minrad, maxrad, longlat_thresh2,
                                        rad_thresh, template_thresh, target_thresh, rw)
        else:
            print('Found craters: ' + str(len(templ_coords)))
            self.logger.info('Found craters: ' + str(len(templ_coords)))

        # find max detected crater radius
        maxr = 0
        if len(templ_coords > 0):
            maxr = np.max(templ_coords.T[2])

        # compare template-matched results to ground truth csv input data
        N_match = 0
        frac_dupes = 0
        err_lo, err_la, err_r = 0, 0, 0
        N_csv, N_detect = len(csv_coords), len(templ_coords)
        for lo, la, r in templ_coords:
            Long, Lat, Rad = csv_coords.T
            minr = np.minimum(r, Rad)

            dL = ((Long - lo)**2 + (Lat - la)**2) / minr**2
            dR = abs(Rad - r) / minr
            index = (dR < rad_thresh) & (dL < longlat_thresh2)
            index_True = np.where(index == True)[0]
            N = len(index_True)
            if N >= 1:
                Lo, La, R = csv_coords[index_True[0]].T
                meanr = (R + r) / 2.
                err_lo += abs(Lo - lo) / meanr
                err_la += abs(La - la) / meanr
                err_r += abs(R - r) / meanr
                if N > 1: # duplicate entries hurt recall
                    frac_dupes += (N-1) / float(len(templ_coords))
            N_match += min(1, N)
            # remove csv(s) so it can't be re-matched again
            csv_coords = csv_coords[np.where(index == False)]
            if len(csv_coords) == 0:
                break

        if rmv_oor_csvs == 1:
            upper = 15
            lower = minrad_
            N_large_unmatched = len(np.where((csv_coords.T[2] > upper) |
                                             (csv_coords.T[2] < lower))[0])
            if N_large_unmatched < N_csv:
                N_csv -= N_large_unmatched

        if N_match >= 1:
            err_lo = err_lo / N_match
            err_la = err_la / N_match
            err_r = err_r / N_match

        stats = [N_match, N_csv, N_detect, maxr]
        #self.logger.info('N_match')
        err = [err_lo, err_la, err_r]
        return stats, err, frac_dupes, templ_coords



    
def get_subset_ha(csv_arr_px, minr_px=6, maxr_px=140):
    #https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
    csv_sub = np.copy(csv_arr_px)
    np.sort(csv_sub, axis=0)
    
def get_time():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    return now

def grab_files_list(path, verbose):
    arr = glob.glob(path) #(glob.glob("Robbins_Dataset/synth_out/extracted_04-16km_on_blur2/*.png"))
    arr.sort()
    if (verbose):
        print(len(arr))
        print(arr)
    return arr

def loadIm(fname, tname, data, target, step=512, newpx = 512, px = 2048):    
    im = plt.imread(fname)
    #px = 2048 #im.size #TODO: FIX THIS IT WILL BREAK EVERYTHING
    print('max: ' + str(im.max()) + ', min: ' + str(im.min()) + ', mean: ' + str(im.mean()))
    tim = 1*(plt.imread(tname)>0) #makes values of target binary
    counter = 0
    print(im.shape)
    print(tim.shape)
    for y in range(0,px,step): #no need to sub 512 b/c px are mult of 512
        for x in range(0,px,step):
            data.append(im[x:x+newpx,y:y+newpx].reshape((newpx,newpx,1)))
            target.append(tim[x:x+newpx,y:y+newpx].reshape((newpx,newpx,1)))

def remake_tile(images, tile_size=7680, stp=512, SAVE=False, SHOW=False, img_fn=None):
    #figure out the grid size
    num_images = len(images)
    grid_size = int(np.sqrt(num_images))
    #stp = 512
    
    #make list of coordinates 
    coords = []
    for x in range(grid_size):
        for y in range(grid_size):
            coords.append([x*stp, y*stp])
    grid_tile = np.zeros((tile_size,tile_size))

    #place each subtile in the larger tile
    for i, im in enumerate(images):
        grid_tile[coords[i][1]:coords[i][1]+stp,coords[i][0]:coords[i][0]+stp] = im[:,:,0]
        
    if(SHOW):
        plt.imshow(grid_tile)
        plt.gcf().set_size_inches((12,12))
        plt.show()
    
    if(SAVE and img_fn is not None):
        plt.imsave(img_fn+'.png',grid_tile)
        
    return grid_tile 

#MAKE PICTURE FROM CRATER LIST
def crater_list_to_image(crater_array, img_size=2048):
    craters_found_img = np.zeros((img_size,img_size))
    for i in range(len(crater_array)):
        x_ctr = crater_array[i][0]; y_ctr = crater_array[i][1]; r=crater_array[i][2]
        brightness = 255; thick = 4
        #cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
        cv2.circle(craters_found_img,(x_ctr,y_ctr), r, brightness, thick) 
        #print(x_ctr)
    
    plt.gcf().set_size_inches((12,12))
    plt.imshow(craters_found_img)
    plt.show()
    return craters_found_img

def four_image(data_image, targ_image, pred_image, find_image, start_x=0, 
               start_y=0, wid_ht=1024, img_fn=None, SAVE=False, SHOW=True):
    #Show Subset of Tile
    sx=start_x; sy=start_y; swh=wid_ht
    
    plt.subplot(2,2,1)
    plt.title('Data')
    plt.imshow(data_image[sx:sx+swh, sy:sy+swh])

    plt.subplot(2,2,2)
    plt.title('Target')
    plt.imshow(targ_image[sx:sx+swh, sy:sy+swh])

    plt.subplot(2,2,3)
    plt.title('NN Prediction')
    plt.colorbar()
    plt.imshow(pred_image[sx:sx+swh, sy:sy+swh])

    plt.subplot(2,2,4)
    plt.title('Crater Finder Output')
    #plt.colorbar()
    plt.imshow(find_image[sx:sx+swh, sy:sy+swh])

    plt.gcf().set_size_inches((12,12))
    
    if(SAVE and img_fn is not None):
        plt.imsave(img_fn+'.png',grid_tile)
    
    if(SHOW):
        plt.show()
    
#Make the csv px array, pull columns 3-5, reorder
def make_csv_px_array(csv_px_fn):
    tile_csv = pd.read_csv(csv_px_fn)
    tile_csv_px = tile_csv.as_matrix(columns=tile_csv.columns[3:6]) #numpy array
    print(tile_csv_px)

    tile_csv_px_xyr = np.copy(tile_csv_px) #making a copy isn't strictly necessary

    #switch order of first two cols of new array from y-x-rad to x-y-rad
    tile_csv_px_xyr[:,[0, 1, 2]] = tile_csv_px_xyr[:,[1, 0, 2]]

    print(tile_csv_px_xyr)
    return tile_csv_px_xyr

def make_comparison_plot(img_fn, coords, csv_px_xyr, rpx_min=7.9, rpx_max=138.2, save_fn=None, SAVE=True, SHOW=False):
    #load grayscale image, cv2 loads as color by default
    #img = np.zeros((7680,7680,3), np.uint8) #start with black, color image
    img = cv2.imread(img_fn) #default loads as color image even though grayscale

    #make a copy of the numpy arrays
    crater_array = np.copy(coords)
    from_csv = np.copy(csv_px_xyr)

    #Add All the Annotation Craters
    counter = 0 #counter will be the number of craters within the px range
    for i in range(len(from_csv)):
        x_ctr = from_csv[i][0]; y_ctr = from_csv[i][1]; r=from_csv[i][2]
        brightness = 255; thick = 8
        #cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
        if(r<rpx_max and r>rpx_min):
            #annotation craters in blue
            cv2.circle(img,(x_ctr,y_ctr), r, (0,0,255), thick) #blue
            counter=counter+1
    print(counter)

    for i in range(len(crater_array)): #found craters
        x_ctr = crater_array[i][0]; y_ctr = crater_array[i][1]; r=crater_array[i][2]
        brightness = 255; thick = 8
        #cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
        #found craters in green
        cv2.circle(img,(x_ctr,y_ctr), r, (0,255,0), int(thick/2)) #green

    #if (SAVE is True and save_fn is not None):
    #    print('Saving file at: ' + save_fn + '.png')
    #    cv2.imwrite(save_fn + '.png', img) #GIANT file >100 MB
    
    if(SHOW or SAVE):
        plt.imshow(img)
        plt.gcf().set_size_inches((12,12))
        plt.xticks([]), plt.yticks([])
        if (SAVE):
            plt.savefig(save_fn + '.png')
        if (SHOW):
            plt.show()

        plt.imshow(img[0:2048,0:2048,:])
        plt.gcf().set_size_inches((12,12))
        plt.xticks([]), plt.yticks([])
        #plt.savefig(save_fn + '_zoom' + '.png')
        #plt.show()
        if (SAVE):
            plt.savefig(save_fn + '_zoom' + '.png')
        if (SHOW):
            plt.show()
        
    return counter

def run_all_tiles(mt):
    for i in range(24):
        mt.logger.info('Starting processing for Tile '+"{:02}".format(i))
        mt.logger.info('CSV, human annotated: '+ mt.csv_hu_arr[i])
        mt.logger.info('Data:                  '+ mt.data_arr[i])
        mt.logger.info('Target:                '+ mt.targ_arr[i])

        print('\n\n\n\n')
        print(mt.csv_hu_arr[i], mt.data_arr[i], mt.targ_arr[i], '\n', sep=' \n ')
        data_fn = mt.data_arr[i]     #'Robbins_Dataset/out/thm_dir_N-30_090_-30_0_90_120_filled.png'
        targ_fn = mt.targ_arr[i]     #'Robbins_Dataset/out/thm_dir_N-30_090_-30_0_90_120_2_32_km_segrng_8_edge.png'
        csv_px_fn = mt.csv_hu_arr[i] #'Robbins_Dataset/csv/LatLonDiam_RobbinsCraters_20121016_-30_0_90_120_px.csv'

        tile_pred, coords = mt.run_match_one_tile(data_fn, targ_fn, csv_px_fn)
        stats, err, frac_dupes, templ_coords, csv_px_xyr = mt.run_compare_one_tile(csv_px_fn, tile_pred, coords)

        sv_fn = 'plots/found/Tile_'+"{:02}".format(i)+'_'+mt.version+'_'+get_time()+'_match_comparison'
        craters_in_range = make_comparison_plot(data_fn, coords, csv_px_xyr, save_fn=sv_fn)
        mt.logger.info('Saved comparison plot: '+ sv_fn)
        
        print('Matches Ratio (matches/craters_in_range): ' + str(stats[0]/craters_in_range))
        mt.logger.info('Tile ' + "{:02}".format(i) + ' Matches: ' + str(stats[0]))
        mt.logger.info('Tile ' + "{:02}".format(i) + ' Craters_in_range: ' + str(craters_in_range))
        mt.logger.info('Tile ' + "{:02}".format(i) + ' Matches ratio (matches/craters_in_range): ' + 
                       str(stats[0]/craters_in_range))
        
        print('Done at: ' + get_time())
        mt.logger.info('Done with Tile '+"{:02}".format(i))
        mt.logger.info(' ')
        print('\n\n\n\n')
        
def run_some_tiles(mt, run_list):
    mt.logger.info('Running SOME tiles: ' + str(run_list))
    
    for i in run_list:
        mt.logger.info('Starting processing for Tile '+"{:02}".format(i))
        mt.logger.info('CSV, human annotated: '+ mt.csv_hu_arr[i])
        mt.logger.info('Data:                  '+ mt.data_arr[i])
        mt.logger.info('Target:                '+ mt.targ_arr[i])

        print('\n\n\n\n')
        print(mt.csv_hu_arr[i], mt.data_arr[i], mt.targ_arr[i], '\n', sep=' \n ')
        data_fn = mt.data_arr[i]     #'Robbins_Dataset/out/thm_dir_N-30_090_-30_0_90_120_filled.png'
        targ_fn = mt.targ_arr[i]     #'Robbins_Dataset/out/thm_dir_N-30_090_-30_0_90_120_2_32_km_segrng_8_edge.png'
        csv_px_fn = mt.csv_hu_arr[i] #'Robbins_Dataset/csv/LatLonDiam_RobbinsCraters_20121016_-30_0_90_120_px.csv'

        tile_pred, coords = mt.run_match_one_tile(data_fn, targ_fn, csv_px_fn)
        stats, err, frac_dupes, templ_coords, csv_px_xyr = mt.run_compare_one_tile(csv_px_fn, tile_pred, coords)

        sv_fn = 'plots/found/Tile_'+"{:02}".format(i)+'_'+mt.version+'_'+get_time()+'_match_comparison'
        craters_in_range = make_comparison_plot(data_fn, coords, csv_px_xyr, save_fn=sv_fn)
        mt.logger.info('Saved comparison plot: '+ sv_fn)
        
        print('Matches Ratio (matches/craters_in_range): ' + str(stats[0]/craters_in_range))
        mt.logger.info('Tile ' + "{:02}".format(i) + ' Matches: ' + str(stats[0]))
        mt.logger.info('Tile ' + "{:02}".format(i) + ' Craters_in_range: ' + str(craters_in_range))
        mt.logger.info('Tile ' + "{:02}".format(i) + ' Matches ratio (matches/craters_in_range): ' + 
                       str(stats[0]/craters_in_range))
        
        print('Done at: ' + get_time())
        mt.logger.info('Done with Tile '+"{:02}".format(i))
        mt.logger.info(' ')
        print('\n\n\n\n')
        

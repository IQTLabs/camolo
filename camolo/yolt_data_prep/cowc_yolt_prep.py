#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:32:51 2017

@author: avanetten

Dataset located at:
    http://gdo-datasci.ucllnl.org/cowc/

"""

import os
import cv2
import sys
import time
import glob
import shutil
import pickle
import shapely
import shapely.geometry
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# path = '/cosmiq/yolt2/scripts'
# sys.path.append(path)
import prep_train
import slice_im
import tile_ims_labels
import prep_train

###############################################################################
def gt_boxes_from_cowc_png(gt_c, yolt_box_size, verbose=False):

    '''
    Get ground truth locations from cowc ground_truth image
    input:
        gt_c is cowc label image
        yolt_box_size is the size of each car in pixels
    outputs:
        box_coords = [x0, x1, y0, y1]
        yolt_coords = convert.conver(box_coords)
    '''
        
    win_h, win_w = gt_c.shape[:2]

    # find locations of labels (locs => (h, w))
    label_locs = list(zip(*np.where(gt_c > 0)))
    
    # skip if label_locs is empty
    if len(label_locs) == 0:
        if verbose:
            print("Label empty")
        return [], []    
                
    if verbose:
        print("Num cars:", len(label_locs))
        
    # else, create yolt labels from car locations
    # make boxes around cars
    box_coords = []
    yolt_coords = []
    grid_half = yolt_box_size/2
    for i,l in enumerate(label_locs):
        
        if verbose and (i % 100) == 0:
            print(i, "/", len(label_locs))
            
        ymid, xmid = l
        x0, y0, x1, y1 = xmid - grid_half, ymid - grid_half, \
                         xmid + grid_half, ymid + grid_half
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(x1, gt_c.shape[1]-1)
        y1 = min(y1, gt_c.shape[0]-1)
        box_i = [x0, x1, y0, y1]
        box_coords.append(box_i)
        # Input to convert: image size: (w,h), box: [x0, x1, y0, y1]
        yolt_co_i = prep_train.convert((win_w, win_h), box_i)
        yolt_coords.append(yolt_co_i)

    box_coords = np.array(box_coords)
    yolt_coords = np.array(yolt_coords)     

    return box_coords, yolt_coords     

###############################################################################
def gt_dic_from_box_coords(box_coords):
    '''
    box_coords are of form:
        box_coords = [x0, x1, y0, y1]
    output should be of form:
    x1l0, y1l0 = lineData['pt1X'].astype(int), lineData['pt1Y'].astype(int)
    x2l0, y2l0 = lineData['pt2X'].astype(int), lineData['pt2Y'].astype(int)
    x3l0, y3l0 = lineData['pt3X'].astype(int), lineData['pt3Y'].astype(int)
    x4l0, y4l0 = lineData['pt4X'].astype(int), lineData['pt4Y'].astype(int) 
    assume pt1 is stern, pt2 is bow, pt3 and pt4 give width

    '''
                
    box_coords = np.array(box_coords)
    out_dic = {}

    out_dic['pt1X'] = box_coords[:,0]
    out_dic['pt1Y'] = box_coords[:,2]

    # set p2 as diagonal from p1
    out_dic['pt2X'] = box_coords[:,1] #box_coords[:,1]
    out_dic['pt2Y'] = box_coords[:,3] #box_coords[:,2]

    out_dic['pt3X'] = box_coords[:,1] #box_coords[:,1]
    out_dic['pt3Y'] = box_coords[:,2] #box_coords[:,3]

    out_dic['pt4X'] = box_coords[:,0]
    out_dic['pt4Y'] = box_coords[:,3]

    return out_dic   

###############################################################################
def slice_im_cowc(input_im, input_mask, outname_root, outdir_im, outdir_label, 
             classes_dic, category, yolt_box_size, 
             sliceHeight=256, sliceWidth=256, 
             zero_frac_thresh=0.2, overlap=0.2, pad=0, 
             max_objs_per_slice=1000, verbose=False,
             box_coords_dir='', yolt_coords_dir=''):
    '''
    ADAPTED FROM YOLT/SCRIPTS/SLICE_IM.PY
    Assume input_im is rgb
    Slice large satellite image into smaller pieces, 
    ignore slices with a percentage null greater then zero_fract_thresh'''

    image = cv2.imread(input_im, 1)  # color
    gt_image = cv2.imread(input_mask, 0)
    category_num = classes_dic[category]
    
    im_h, im_w = image.shape[:2]
    win_size = sliceHeight*sliceWidth
    
    # if slice sizes are large than image, pad the edges
    if sliceHeight > im_h:
        pad = sliceHeight - im_h
    if sliceWidth > im_w:
        pad = max(pad, sliceWidth - im_w)
    # pad the edge of the image with black pixels
    if pad > 0:    
        border_color = (0,0,0)
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, 
                                 cv2.BORDER_CONSTANT, value=border_color)

    t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    for y in range(0, im_h, dy):#sliceHeight):
        for x in range(0, im_w, dx):#sliceWidth):
            n_ims += 1
            # extract image
            # make sure we don't go past the edge of the image
            if y + sliceHeight > im_h:
                y0 = im_h - sliceHeight
            else:
                y0 = y
            if x + sliceWidth > im_w:
                x0 = im_w - sliceWidth
            else:
                x0 = x
            
            window_c = image[y0:y0 + sliceHeight, x0:x0 + sliceWidth]
            gt_c = gt_image[y0:y0 + sliceHeight, x0:x0 + sliceWidth]
            win_h, win_w = window_c.shape[:2]
            
            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)

            # find threshold of image that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret,thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            #print("zero_frac", zero_fra   
            # skip if image is mostly empty
            if zero_frac >= zero_frac_thresh:
                if verbose:
                    print("Zero frac too high at:", zero_frac)
                continue 
            
#            # find locations of labels (locs => (h, w))
#            label_locs = zip(*np.where(gt_c > 0))
#            
#            # skip if label_locs is empty
#            if len(label_locs) == 0:
#                if verbose:
#                    print("Label file emty for slice"
#                continue     
                        
#            # else, create yolt labels from car locations
#            # make boxes around cars
#            box_coords = []
#            yolt_coords = []
#            grid_half = yolt_box_size/2
#            for l in label_locs:
#                ymid, xmid = l
#                x0, y0, x1, y1 = xmid - grid_half, ymid - grid_half, \
#                                 xmid + grid_half, ymid + grid_half
#                x0 = max(0, x0)
#                y0 = max(0, y0)
#                x1 = min(x1, gt_c.shape[1]-1)
#                y1 = min(y1, gt_c.shape[0]-1)
#                box_i = [x0, x1, y0, y1]
#                box_coords.append(box_i)
#                # Input to convert: image size: (w,h), box: [x0, x1, y0, y1]
#                yolt_co_i = convert.convert((win_w, win_h), box_i)
#                yolt_coords.append(yolt_co_i)
#
#            box_coords = np.array(box_coords)
#            yolt_coords = np.array(yolt_coords)          

            box_coords, yolt_coords = gt_boxes_from_cowc_png(gt_c, 
                                                             yolt_box_size, 
                                                             verbose=verbose)
            # skip if no coords
            if len(box_coords) == 0:
                continue
            
            # skip if too many objects
            if len(box_coords) > max_objs_per_slice:
                continue
                
            #  save          
            outname_part = 'slice_' + outname_root + \
            '_' + str(y0) + '_' + str(x0) + '_' + str(win_h) + '_' + str(win_w) +\
            '_' + str(pad)
            outname_im = os.path.join(outdir_im, outname_part + '.png')
            txt_outpath = os.path.join(outdir_label, outname_part + '.txt')
            
            # save yolt ims
            if verbose:
                print("image output:", outname_im)
            cv2.imwrite(outname_im, window_c)
            
            # save yolt labels
            txt_outfile = open(txt_outpath, "w")
            if verbose:
                print("txt output:" + txt_outpath)
            for bb in yolt_coords:
                outstring = str(category_num) + " " + " ".join([str(a) for a in bb]) + '\n'
                if verbose:
                    print("outstring:", outstring)
                txt_outfile.write(outstring)
            txt_outfile.close()

            # if desired, save coords files
            # save box coords dictionary so that yolt_eval.py can read it                                
            if len(box_coords_dir) > 0: 
                coords_dic = gt_dic_from_box_coords(box_coords)
                outname_pkl = os.path.join(box_coords_dir, outname_part + '_' + category + '.pkl')
                pickle.dump(coords_dic, open(outname_pkl, 'wb'), protocol=2)
            if len(yolt_coords_dir) > 0:  
                outname_pkl = os.path.join(yolt_coords_dir, outname_part + '_' + category + '.pkl')
                pickle.dump(yolt_coords, open(outname_pkl, 'wb'), protocol=2)

            n_ims_nonull += 1

    print("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull, \
            "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
    print("Time to slice", input_im, time.time()-t0, "seconds")
      
    return

###############################################################################
def rescale_ims(indir, outdir, inGSD, outGSD, resize_orig=False):
    '''
    take images in indir and rescale them to the appropriate GSD
    assume inputs are square
    if resize_orig, rescale downsampled image up to original image size
    
    A properly designed sensor should have pixels size determined by the 
    Nyquist sampling rate, of half the size of the mirror resolution
    
    '''
    
    t0 = time.time()
    print("indir:", indir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filelist = [f for f in os.listdir(indir) if f.endswith('.png')]
    #filelist = [f for f in os.listdir(indir) if f.endswith('.tif')]
    lenf = len(filelist)
    for i,f in enumerate(filelist):
        if (i % 100) == 0:
            print(i, "/", lenf)
        ftot = indir + f
        # load image
        img_in = cv2.imread(ftot, 1)
            
        # set kernel, multiply by 0.5 to match the Nyquist sampling rate
        kernel = 0.5 * outGSD/inGSD #int(round(blur_meters/GSD_in))
        img_out = cv2.GaussianBlur(img_in, (0, 0), kernel, kernel, 0)
        
        # may want to rescale?
        # reshape, assume that the pixel density is double the point spread
        # function sigma value
        # use INTER_AREA interpolation function
        rescale_frac = inGSD / outGSD
        rescale_shape = int( np.rint(img_in.shape[0] * rescale_frac) ) # / kernel)# * 0.5)# * 2
        #print("rescale_shape:", rescale_shape
        #print("f", f, "kernel", kernel, "shape_in", img_in.shape[0], "shape_out", rescale_shape

        # resize to the appropriate number of pixels for the given GSD
        img_out = cv2.resize(img_out, (rescale_shape,rescale_shape), interpolation=cv2.INTER_AREA)

        if resize_orig:
            # scale back up to original size (useful for length calculations, but
            #   keep pixelization)
            img_out = cv2.resize(img_out, (img_in.shape[1], img_in.shape[0]), interpolation=cv2.INTER_LINEAR)#cv2.INTER_NEAREST)
        
        # write to file
        outf = outdir + f
        #print("outf:", outf
        #outf = fold_tot + f.split('.')[0] + '_blur' + str(blur_meters) + 'm.' + f.split('.')[1] 
        cv2.imwrite(outf, img_out)

    print("Time to rescale", lenf, "images from", indir, inGSD, "GSD, to", \
            outdir, outGSD, "=", time.time() - t0, "seconds")
    return


################################################################################        
#def plot_training_bboxes(label_folder, image_folder, ignore_augment=True,
#                         figsize=(10,10), color=(0,0,255), thickness=2, 
#                         max_plots=100, sample_label_vis_dir=None, ext='.png',
#                         im_out_size=()):
#    '''
#    FROM YOLT_DATA_PREP.PY
#    Plot bounding boxes'''
#    
#    # boats, boats_harbor, airplanes, airports (blue, green, red, orange)
#    # remember opencv uses bgr, not rgb
#    colors = [(255,0,0), (0,255,0), (0,0,255), (0,140,255), (125, 125, 0)]  
#    
#    if sample_label_vis_dir and not os.path.exists(sample_label_vis_dir):
#        os.mkdir(sample_label_vis_dir)
#              
#    cv2.destroyAllWindows()
#    i = 0
#    for label_file in ranos.listdir(label_folder):
#        
#        print i, "/", max_plots
#        
#        if ignore_augment:
#            if (label_file == '.DS_Store') or (label_file.endswith(('_lr.txt', '_ud.txt', '_lrud.txt'))):
#                continue
#        else:
#             if (label_file == '.DS_Store'):
#                 continue
#        i+=1
#        if i >= max_plots:
#            return
#            
#        # get image
#        print("label loc:", label_file
#        root = label_file.split('.')[0]
#        im_loc = image_folder + root + ext
#        print("im_loc:", im_loc
#        
#        image0 = cv2.imread(im_loc, 1)
#        # resize, if desired
#        if len(im_out_size) > 0:
#            image = cv2.resize(image0, im_out_size, 
#                               interpolation=cv2.INTER_AREA)
#        else:
#            image = image0
#        
#        height, width = image.shape[:2]
#        shape = (width, height)
#        
#        ## start plot (mpl)
#        #fig, ax = plt.subplots(figsize=figsize)
#        #img_mpl = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        #ax.imshow(img_mpl)
#        # just opencv
#        img_mpl = image
#        
#        # get and plot labels
#        z = pd.read_csv(label_folder + label_file, sep = ' ', names=['cat', 'x', 'y', 'w', 'h'])
#        #print("z", z.values
#        for yolt_box in z.values:
#            cat_int = int(yolt_box[0])
#            color = colors[cat_int]
#            yb = yolt_box[1:]
#            box0 = convert.convert_reverse(shape, yb)
#            # convert to int
#            box1 = [int(round(b,2)) for b in box0]
#            [xmin, xmax, ymin, ymax] = box1
#            # plot
#            cv2.rectangle(img_mpl, (xmin, ymin), (xmax, ymax), (color), thickness)    
#
#        #cv2.waitKey(0)
#        
#        if sample_label_vis_dir:
#            fout = sample_label_vis_dir + root + '_vis.png'
#            cv2.imwrite(fout, img_mpl)
#
#    return

###############################################################################        
def plot_gt_boxes(im_file, label_file, yolt_box_size,
                  figsize=(10,10), color=(0,0,255), thickness=2):
    '''
    plot ground truth boxes overlaid on image
    '''
    
    
    im = cv2.imread(im_file)
    gt_c = cv2.imread(label_file, 0)
    box_coords, yolt_coords = gt_boxes_from_cowc_png(gt_c, yolt_box_size,
                                                     verbose=False)
    
    img_mpl = im
    for b in box_coords:
        [xmin, xmax, ymin, ymax] = b

        cv2.rectangle(img_mpl, (xmin, ymin), (xmax, ymax), (color), thickness)    

###############################################################################
def box_coords_to_gdf(box_coords, image_path, category):
    '''Convert box_coords to geodataframe, assume schema:      
        box_coords = [x0, x1, y0, y1]
        Adapted from parse_shapefile.py'''
        
    pix_geom_poly_list = []
    for b in box_coords:
        [x0, x1, y0, y1] = b
        out_coords = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
        points = [shapely.geometry.Point(coord) for coord in out_coords]
        pix_poly = shapely.geometry.Polygon([[p.x, p.y] for p in points])
        pix_geom_poly_list.append(pix_poly)

    df_shp = pd.DataFrame(pix_geom_poly_list, columns=['geometry_poly_pixel'])
                
    #df_shp['geometry_poly_pixel'] = pix_geom_poly_list
    df_shp['geometry_pixel'] = pix_geom_poly_list
    df_shp['shp_file'] = ''
    df_shp['Category'] = category
    df_shp['Image_Path'] = image_path
    df_shp['Image_Root'] = image_path.split('/')[-1]
    
    return df_shp
    

###############################################################################
def rescale_cowc_annotations(scaled_image_dir, annotated_image_dir, out_dir,
                             annotation_suffix='_Annotated_Cars.png',
                             verbose=False):
    '''Rescale annotation files from already rescaled images
    DOESN'T WORK CORRECTLY AS THERE ARE A DIFFERENT NUMBER OF NONZERO PIXELS!
    '''
    
    im_list = [z for z in os.listdir(scaled_image_dir) if z.endswith('.png')]
    for im_file in im_list:
        im_file_root, ext = im_file.split('.')
        im_file_loc = os.path.join(scaled_image_dir, im_file)
        label_file_loc = os.path.join(annotated_image_dir, im_file_root 
                                              + annotation_suffix)
        out_file = os.path.join(out_dir, im_file_root + annotation_suffix)
        
        print("im_file:", im_file)
        print("im_file_loc:", im_file_loc)
        print("label_file_loc:", label_file_loc)
        print("os.path.exists(im_file_loc)?:", os.path.exists(im_file_loc))
        print("os.path.exists(label_file_loc):", os.path.exists(label_file_loc))
        
        # read in image
        im = cv2.imread(im_file_loc, 0)
        h,w = im.shape[:2]
        print("im.shape:", im.shape)
        
        # resize label_im
        label_im = cv2.imread(label_file_loc, 0)
        print("label_im.shape:", label_im.shape)
        print("Number of input nonzero pixels", len(np.where(label_im > 0)[0]))
        img_out = cv2.resize(label_im, (w, h), interpolation=cv2.INTER_AREA)
        print("Number of output nonzero pixels", len(np.where(img_out > 0)[0]))

        cv2.imwrite(out_file, img_out)

    return
    

#%% COWC (Cars from LLNL)
#######################################################################
#######################################################################
# see cowc_explore.py to explore dataset
def main():
    
    ##########################
    yolt_dir = '/cosmiq/yolt2/'
    cowc_dir = '/cosmiq/cowc/'
    output_loc = '/raid/local/src/yolt2/training_datasets/'  # eventual location of files 
   
    ###############
    # cars
    classes_dic = {"car": 5}    
    slice_overlap = 0.1
    zero_frac_thresh = 0.2
    #sliceHeight, sliceWidth = 416, 416
    #sliceHeight, sliceWidth = 960, 960  # for for 167 m windows
    sliceHeight, sliceWidth = 832, 832 # for for 124m windows

    train_name = 'cars_'  + str(sliceHeight)
    train_base_dir = yolt_dir + 'training_datasets/' + train_name + '/'


    ################
    # infer variables from previous settings
    #input_images_raw_dir = train_base_dir + 'ims_input_raw/'
    #split_dir = train_base_dir + train_name + '_split/'
    t_dir = train_base_dir + 'training_data/'
    labels_dir = t_dir + 'labels/'
    images_dir = t_dir + 'images/'
    train_images_list_file_loc = yolt_dir + 'data/'
    im_list_name = train_base_dir + train_name + '_list.txt'
    im_locs_for_list = output_loc + train_name + '/' + 'training_data/images/'
    sample_label_vis_dir = train_base_dir + 'sample_label_vis/'

    # make directories
    for tmp_dir in [train_base_dir, t_dir, labels_dir,
                    images_dir, sample_label_vis_dir, 
                    #input_images_raw_dir, split_dir
                    ]:
        print("dir:", tmp_dir)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

    # load scripts
    os.chdir(yolt_dir)
    sys.path.append(yolt_dir + 'scripts')
    import convert, slice_im
    reload(convert)
    reload(slice_im)

    ##########################
    # Labeling and bonunding box settings
    # put all images in yolt/images/boat
    # put all labels in yolt/labels/boat
    # put list of images in yolt/darknet/data/boat_list2_dev_box.txt
    ##########################    

    #%% SLICE training images (DON'T DO FOR RE-ANALYSIS!)    
    run_slice = False     # switch to actually slice up the raw input images
    
    # dictionary of train_test
    # for now skip Columbus and Vahingen since they are grayscale
    ground_truth_dir = cowc_dir + 'datasets/ground_truth_sets/'
    train_dirs = ['Potsdam_ISPRS', 'Selwyn_LINZ', 'Toronto_ISPRS']
    test_dirs = ['Utah_AGRC']
    annotation_suffix = '_Annotated_Cars.png'
    category = 'car'
    category_num = classes_dic[category]

    # set yolt training box size
    car_size = 3      # meters
    GSD = 0.15          # meters
    yolt_box_size = np.rint(car_size/GSD)
    print("yolt_box_size:", yolt_box_size)
  

    ## test
    #gt_c_test = cv2.imread('/cosmiq/cowc/datasets/ground_truth_sets/Potsdam_ISPRS/top_potsdam_2_10_RGB_Annotated_Cars.png', 0)
    #box_coords_tst, yolt_coords_tst = gt_boxes_from_cowc_png(gt_c_test, yolt_box_size, 
    #                                                         verbose=False)

    for i,d in enumerate(train_dirs):
        dtot = ground_truth_dir + d + '/'
        print("dtot:", dtot)

        # get label files
        files = os.listdir(dtot)
        annotate_files = [f for f in files if f.endswith(annotation_suffix)]
        #print("annotate_files:", annotate_files
    
        for annotate_file in annotate_files:
            annotate_file_tot = dtot + annotate_file
            name_root = annotate_file.split(annotation_suffix)[0]
            imfile = name_root + '.png'
            imfile_tot = dtot + imfile
            outroot = d + '_' + imfile.split('.')[0]
            print("\nName_root", name_root)
            #print("annotate_file:", annotate_file
            #print("  imfile:", imfile
            #print("  imfile_tot:", imfile_tot
            print("  outroot:", outroot)
        
            if run_slice:
                slice_im_cowc(imfile_tot, annotate_file_tot, outroot, images_dir, 
                         labels_dir, classes_dic, category, yolt_box_size, 
                         sliceHeight=sliceHeight, sliceWidth=sliceWidth, 
                         zero_frac_thresh=zero_frac_thresh, overlap=slice_overlap, 
                         pad=0, verbose=False)

       
    #%% plot box labels
    max_plots=50
    thickness=1
    im_out_size=(416,416)
    #for res in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.2, 1.8, 2.4, 3.0]:
    for res in [0.3]:
        out_res_str = str(res).replace('.', 'p')
        suff = '_' + out_res_str + 'GSD/'
        images_dir_tmp = images_dir[:-1] + suff
        sample_label_vis_dir_tmp = sample_label_vis_dir[:-1] + suff
        print("images_dir_tmp:", images_dir_tmp)
        print("sample_label_vis_dir_tmp:", sample_label_vis_dir_tmp)
        yolt_data_prep_funcs.plot_training_bboxes(labels_dir, images_dir_tmp, ignore_augment=False,
                         sample_label_vis_dir=sample_label_vis_dir_tmp, 
                         max_plots=max_plots, thickness=thickness, ext='.png')

    #plt.close("all")
        

    #%% get list for yolt/data/, copy to data dir
    train_ims = [im_locs_for_list + f for f in  os.listdir(images_dir)]
    f = open(im_list_name, 'wb')
    for item in train_ims:
        f.write("%s\n" % item)
    f.close()
    shutil.copy(im_list_name, train_images_list_file_loc)

  
    #%% downsample training data
    inGSD = 0.15
    #GSD_list = [0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    GSD_list = [0.15]

    indir = images_dir

    for outGSD in GSD_list:
        out_res_str = str(outGSD).replace('.', 'p')
        suffix = '_' + out_res_str + 'GSD'
        #suffix = '_' + str(outGSD) + 'GSD'    
        outdir = indir[:-1] + suffix + '/'
        print("outdir:", outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        rescale_ims(indir, outdir, inGSD, outGSD, resize_orig=False)
    
        # copy labels to outdir_labels
        outdir_labels = labels_dir[:-1] + suffix + '/'
        if not os.path.exists(outdir_labels):
            os.mkdir(outdir_labels)
        for filename in glob.glob(os.path.join(labels_dir, '*.*')):
            shutil.copy(filename, outdir_labels)
    
        # get list for yolt/data/, copy to data dir
        im_list_name =  train_base_dir + train_name + suffix + '_list.txt'
        im_locs_for_list_tmp = im_locs_for_list[:-1] + suffix + '/'
        #train_ims = [im_locs_for_list_tmp + ftmp for ftmp in os.listdir(outdir)]
        train_ims = [os.path.join(outdir, ftmp) for ftmp in os.listdir(outdir)]
        print("train_ims[:5]:", train_ims[:5])
        f = open(im_list_name, 'wb')
        for item in train_ims:
            f.write("%s\n" % item)
        f.close()
        shutil.copy(im_list_name, train_images_list_file_loc)
    
    #%% Augment (only do this once, or it expands the dataset...)
    augment = False
    ###############################################################################
    image_folder_tmp = '/cosmiq/yolt2/training_datasets/cars_832/training_data/images_0p3GSD'
    label_folder_tmp = '/cosmiq/yolt2/training_datasets/cars_832/training_data/labels_0p3GSD'
    if augment:
        yolt_data_prep_funcs.augment_training_data(label_folder_tmp, image_folder_tmp, hsv_range=[0.5,1.5],
                              skip_hsv_transform=False, ext='.png')
    
    #%% New file list
    output_loc = '/raid/local/src/yolt2/'  # eventual location of files 
    #suffix = '_' + str(outGSD) + 'GSD'    
    out_res_str = str(res).replace('.', 'p')
    suffix = '_' + out_res_str + 'GSD'


    augment_ims = os.listdir(image_folder_tmp)
    # update image list
    if augment:
        im_list_loc_tmp0 = train_base_dir + train_name + '_list' + suffix + '_aug.txt'
        im_list_loc_tmp = train_base_dir + train_name + '_devbox_list' + suffix + '_aug.txt'
    else:
        im_list_loc_tmp0 = train_base_dir + train_name + '_list' + suffix + '.txt'
        im_list_loc_tmp = train_base_dir + train_name + '_devbox_list' + suffix + '.txt'

    f = open(im_list_loc_tmp, 'wb')
    for item0 in augment_ims:
        item1 = os.path.join(image_folder_tmp, item0)
        item = item1.replace(yolt_dir, output_loc)
        f.write("%s\n" % item)
    f.close()
    
    
    #%% plot box labels
    max_plots=50
    thickness=1
    im_out_size=(416,416)
    #for res in [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.2, 1.8, 2.4, 3.0]:
    for res in [0.3]:
        suff = '_' + str(res) + 'GSD/'
        images_dir_tmp = images_dir[:-1] + suff
        labels_dir_tmp = labels_dir[:-1] + suff
        sample_label_vis_dir_tmp = sample_label_vis_dir[:-1] + suff
        print("images_dir_tmp:", images_dir_tmp)
        print("sample_label_vis_dir_tmp:", sample_label_vis_dir_tmp)
        yolt_data_prep_funcs.plot_training_bboxes(labels_dir_tmp, images_dir_tmp, ignore_augment=False,
                         sample_label_vis_dir=sample_label_vis_dir_tmp, 
                         max_plots=max_plots, thickness=thickness, ext='.png',
                         shuffle=True)

    #plt.close("all")
        
    #%% combine 0.15m and 0.30m data
    suffix = '_0.45+0.90GSD'    
    suff1 = '_' + str(0.45) + 'GSD'    
    suff2 = '_' + str(0.9) + 'GSD'    
    im_list_name =  train_base_dir + train_name + suffix + '_list.txt'
    f1 = train_base_dir + train_name + suff1 + '_list.txt'
    f2 = train_base_dir + train_name + suff2 + '_list.txt'
    # combine files
    os.system('cat ' + f1 + ' > ' + im_list_name)
    os.system('cat ' + f2 + ' >> ' + im_list_name)
    shutil.copy(im_list_name, train_images_list_file_loc)

    #%% Convert data to black and white 
    suff = '_' + str(0.3) + 'GSD'
    suff2 = suff + '_pan'
    tmpdir_in = images_dir[:-1] + suff
    tmpdir_out = images_dir[:-1] + suff2
    if not os.path.exists(tmpdir_out): 
        os.mkdir(tmpdir_out)
    for filename in os.listdir(tmpdir_in):
        if not filename.endswith('.png'):
            continue
        #print filename
        im_gray = cv2.imread(os.path.join(tmpdir_in, filename), 0)
        cv2.imwrite(os.path.join(tmpdir_out, filename), im_gray)

    # copy labels too
    labeldir_tmp = labels_dir[:-1] + suff2 
    shutil.copytree(labels_dir[:-1] + suff, labeldir_tmp)

    # get list for yolt/data/, copy to data dir
    im_locs_for_list_tmp = im_locs_for_list[:-1] + suff2 + '/'
    im_list_name_tmp =  train_base_dir + train_name + suff2 + '_list.txt'
    train_ims = [im_locs_for_list_tmp + ftmp for ftmp in  os.listdir(tmpdir_out)]
    f = open(im_list_name_tmp, 'wb')
    for item in train_ims:
        f.write("%s\n" % item)
    f.close()
    shutil.copy(im_list_name_tmp, train_images_list_file_loc)


    ###############################################################################
    #%% TEST DATA
    # turn labels into yolt boxes
    classes_dic = {"car": 4}    
    category = 'car'
    ground_truth_dir = cowc_dir + '/datasets/ground_truth_sets/'
    test_dir = os.path.join(ground_truth_dir, 'Utah_AGRC')
    label_dir = os.path.join(test_dir, 'labels')
    yolt_image_loc = os.path.join(yolt_dir, 'test_images/cowc_utah_raw')
    yolt_label_loc = os.path.join(yolt_dir, 'test_images/cowc_utah_raw/labels')
    verbose = True

    for f in [label_dir, yolt_image_loc, yolt_label_loc]:
        if not os.path.exists(f): 
            os.mkdir(f)
   
    # get label files
    files = os.listdir(test_dir)
    annotate_files = [ftmp for ftmp in files if ftmp.endswith(annotation_suffix)]
    print("\nAnnotate_files:", annotate_files)

    for annotate_file in annotate_files:
        annotate_file_tot = os.path.join(test_dir, annotate_file)
        name_root = annotate_file.split(annotation_suffix)[0]
        imfile = name_root + '.png'
        imfile_tot = os.path.join(test_dir, imfile)
        #outroot = d + '_' + imfile.split('.')[0]
        outroot = imfile.split('.')[0]

        if verbose:
            print("\nannotate_file:", annotate_file)
            print("\annotate_file_tot:", annotate_file_tot)
            print("imfile:", imfile)
            print("outroot:", outroot)
   
        # load mask
        gt_c = cv2.imread(annotate_file_tot, 0)
        # load image
        #img_in = cv2.imread(imfile_tot, 1)
        
        # get coords
        box_coords, yolt_coords = gt_boxes_from_cowc_png(gt_c, 
                                                     yolt_box_size, 
                                                     verbose=verbose)
        # save yolt labels
        txt_outpath = os.path.join(label_dir, outroot + '.txt')
        txt_outfile = open(txt_outpath, "w")
        if verbose:
            print("txt output:" + txt_outpath)
        for bb in yolt_coords:
            outstring = str(classes_dic[category]) + " " + " ".join([str(a) for a in bb]) + '\n'
            #if verbose:
            #    print("outstring:", outstring
            txt_outfile.write(outstring)
        txt_outfile.close()
    
        # copy to yolt_dir
        shutil.copy(imfile_tot, yolt_image_loc)
        shutil.copy(txt_outpath, yolt_label_loc)
    
    #%% Plot a few 
    reload(yolt_data_prep_funcs)
    sample_label_vis_dir = os.path.join(test_dir, 'sample_label_vis_dir')
    yolt_data_prep_funcs.plot_training_bboxes(label_dir, test_dir, ignore_augment=True,
                             max_plots=1, 
                             sample_label_vis_dir=sample_label_vis_dir, ext='.png',
                             verbose=True, show_plot=False, specific_labels=[],
                             label_dic=[], output_width=60000, shuffle=True)

    #%% downsample in yolt test directory
    yolt_image_loc = os.path.join(yolt_dir, 'test_images/cowc_utah_raw')
    ground_truth_dir = cowc_dir + '/datasets/ground_truth_sets/'
    test_dir = os.path.join(ground_truth_dir, 'Utah_AGRC')

    indir = yolt_image_loc
    label_dir = os.path.join(test_dir, 'labels')
    #indir = yolt_dir + 'test_images/cowc_utah_comb/'

    print("indir:", indir)

    inGSD = 0.15
    resize_orig = True#False
    #GSD_list = [0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
    GSD_list = [0.3]
    for outGSD in GSD_list:
        print("\nGSD:", outGSD)
        suffix = '_' + str(outGSD) + 'GSD'    
        #outdir = indir[:-1] + suffix + '/'
        outdir = indir + suffix + '/'
        print("outdir:", outdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        rescale_ims(indir, outdir, inGSD, outGSD, resize_orig=resize_orig)
    
        # copy labels to file
        label_dir_tmp = os.path.join(outdir, 'labels')
        print("label_dir_in:", label_dir)
        print("label_dir_out:", label_dir_tmp)
        if os.path.exists(label_dir_tmp):
            shutil.rmtree(label_dir_tmp)
        shutil.copytree(label_dir, label_dir_tmp)
    
    
    #%% Downsample annotated images
    # DOESN'T WORK PROPERLY; RESCALED IMAGE HAS DIFFERENT NUMBER OF NONZERO PIXELS
    '''
    scaled_image_dir = '/cosmiq/simrdwn/test_images/cowc_utah_raw_0p3GSD'
    annotated_image_dir = '/cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC'
    out_dir = os.path.join(annotated_image_dir, '_rescaled_annotations_0p3GSD')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    rescale_cowc_annotations(scaled_image_dir, annotated_image_dir, out_dir,
                                 annotation_suffix='_Annotated_Cars.png')
    '''

    #%% Save raw ground truth file to geodataframe using yolt labels
    category = 'car'
    indir = yolt_dir + 'test_images/cowc_utah_raw_0p3GSD_rescale'
    label_dir = os.path.join(indir, 'labels')
    df_dir = os.path.join(indir, 'dfs')
    if not os.path.exists(df_dir): os.mkdir(df_dir)
    verbose = True

    test_ims = [f0[:-4] for f0 in os.listdir(indir) if f0.endswith('.png')]
    for i,f in enumerate(test_ims):
        label_loc = os.path.join(label_dir, f + '.txt')
        ftot = os.path.join(indir, f + '.png')
        froot = f.split('.')[0]
        out_df_file = os.path.join(df_dir, froot + '_df.csv')
        out_df_pkl = os.path.join(df_dir, froot + '_df.pkl')

        im = cv2.imread(ftot)
        h,w = im.shape[:2]

        if verbose:
            print("\nfile:", f)
            print("  image_path:", ftot)
            print("  im.shape:", im.shape)
    
        # read labels
        cat_list, box_list = yolt_data_prep_funcs.yolt_labels_to_bbox(label_loc, w, h)
        #print box_list
        #print ftot, label_loc
    
        #if verbose:
        #    print("    cat_list:", cat_list
        #    print("    box_list:", box_list
    
        # save dataframe      
        image_path = ftot
        df = box_coords_to_gdf(box_list, image_path, category)

        df.to_csv(out_df_file) 
        df.to_pickle(out_df_pkl)
        #pickle.dump(df, open(out_df_pkl, 'wb'), protocol=2)
    
        if verbose:
            print("ftot:", ftot)
            print("len(df):", len(df))

        if i == 0:
            df_tot = df
        else:
            df_tot = df_tot.append(df)
    
    # save to file
    df_pkl_tot = os.path.join(df_dir, '00_' + indir.split('/')[-1] + '_df_tot.pkl')
    df_tot.to_pickle(df_pkl_tot)
    print("\nlen(df_tot):", len(df_tot))

    ###########
    ##########


    #%% Save raw ground truth file in correct format (old)
    #   use modular_sliding_window.ground_truth_boxes_func to load, so ground truth
    #   should have format:
    #    x1l0, y1l0 = lineData['pt1X'].astype(int), lineData['pt1Y'].astype(int)
    #    x2l0, y2l0 = lineData['pt2X'].astype(int), lineData['pt2Y'].astype(int)
    #    x3l0, y3l0 = lineData['pt3X'].astype(int), lineData['pt3Y'].astype(int)
    #    x4l0, y4l0 = lineData['pt4X'].astype(int), lineData['pt4Y'].astype(int)    

    category = 'car'
    test_in_dir = ground_truth_dir + 'Utah_AGRC/'#'Vaihingen_ISPRS/'#'Toronto_ISPRS/'#'Selwyn_LINZ/'#'Potsdam_ISPRS/'#'Columbus_CSUAV_AFRL/'#'Utah_AGRC/'
    test_ims = [f0[:-4] for f0 in os.listdir(test_in_dir) if ((f0.endswith('.png')) and (not 'Annotated' in f0))] 
    #test_ims = ['12TVK220980-CROP', '12TVL120100-CROP', '12TVL160640-CROP', 
    #            '12TVL160660-CROP', '12TVL180140', '12TVL200180', 
    #            '12TVL220180-CROP', '12TVL220360-CROP', '12TVL240120']
    tstdir_im = yolt_dir + 'test_images/cowc_utah_raw/'
    if not os.path.exists(tstdir_im): os.mkdir(tstdir_im)
    tstdir_gtpkl = yolt_dir + 'test_images/_ground_truth/'
    save_output=False

    n_tot = 0
    for f in test_ims:
        ftot = test_in_dir + f + '.png'
        print("ftot:", ftot)
    
        froot = f.split('.')[0]
        out_df_file = os.path.join(tstdir_im, froot + '_df.csv')

        label_file = test_in_dir + f + annotation_suffix
        print("label file:", label_file)
        gt_c = cv2.imread(label_file, 0)
        box_coords, yolt_coords = gt_boxes_from_cowc_png(gt_c, yolt_box_size,
                                                         verbose=True)
        print("Num cars in", ftot, len(box_coords))
        n_tot += len(box_coords)
        coords_dic = gt_dic_from_box_coords(box_coords)
        outpkl = tstdir_gtpkl + f.split('.')[0] + '_car.pkl'
        #outpkl = tstdir_gtpkl + 'Utah_AGRC_' + f.split('.')[0] + '.pkl'

        # save dataframe      
        image_path = ftot
        df = box_coords_to_gdf(box_coords, image_path, category)
        df.to_csv(out_df_file) 
    
        if save_output:
            shutil.copy(ftot, tstdir_im)

            # save pickle and copy im to test_dir
            pickle.dump(coords_dic, open(outpkl, 'wb'), protocol=2)
    
            # plot
            vis = cv2.imread(ftot)
            img_mpl = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    
            for b in box_coords:
                [xmin, xmax, ymin, ymax] = b.astype(int)
                cv2.rectangle(img_mpl, (xmin, ymin), (xmax, ymax), (0,0,255), thickness=2)    
        
            #cv2.imshow(f, img_mpl)
            cv2.imwrite(sample_label_vis_dir + '_00_' + f + '.png' , img_mpl)



        
        

    print("Number of cars in folder:", test_in_dir, "=", n_tot)

    #Number of cars in folder: /cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC/ = 19807
    # Num cars in /cosmiq/cowc/datasets/ground_truth_sets/Utah_AGRC/12TVL240120.png 13213
    #Number of cars in folder: /cosmiq/cowc/datasets/ground_truth_sets/Columbus_CSUAV_AFRL/ = 1748
    #Number of cars in folder: /cosmiq/cowc/datasets/ground_truth_sets/Potsdam_ISPRS/ = 2083
    #Number of cars in folder: /cosmiq/cowc/datasets/ground_truth_sets/Selwyn_LINZ/ = 1197
    #Number of cars in folder: /cosmiq/cowc/datasets/ground_truth_sets/Toronto_ISPRS/ = 10023
    #Number of cars in folder: /cosmiq/cowc/datasets/ground_truth_sets/Vaihingen_ISPRS/ = 2863

    #plt.show()
    #%%

    # SLICE TEST DATA IF DESIRED, also save ground truth pkl to correct format
    # slice to 600 meters on a side = 4000 pixels
    run_slice_test = True    
    slice_overlap = 0.025        
    test_dirs = ['Utah_AGRC']
    annotation_suffix = '_Annotated_Cars.png'
    category_num = classes_dic['car']
    out_test = yolt_dir + 'test_images/cowc_utah/'#_0.15GSD/'
    out_test_im = out_test #+ 'images/'
    out_test_label = out_test + 'labels/'
    #out_test_box_coords = out_test + 'box_coords/'
    out_test_box_coords = yolt_dir + 'test_images/_ground_truth/'
    category = 'car'

    sliceHeight, sliceWidth = 4000, 4000

    if run_slice_test:
        for d in [out_test, out_test_im, out_test_label, out_test_box_coords]:
            if not os.path.exists(d):
                os.mkdir(d)
      
        for i,d in enumerate(test_dirs):
            dtot = ground_truth_dir + d + '/'
            print("dtot:", dtot)
    
            # get label files
            files = os.listdir(dtot)
            annotate_files = [f for f in files if f.endswith(annotation_suffix)]
            #print("annotate_files:", annotate_files
        
            for annotate_file in annotate_files:
                annotate_file_tot = dtot + annotate_file
                name_root = annotate_file.split(annotation_suffix)[0]
                imfile = name_root + '.png'
                imfile_tot = dtot + imfile
                #outpkl_box = out_test_box_coords + d + '_' + name_root + '.pkl'

                outroot = d + '_' + imfile.split('.')[0]
                print("\nName_root", name_root)
                #print("annotate_file:", annotate_file
                #print("  imfile:", imfile
                #print("  imfile_tot:", imfile_tot
                print("  outroot:", outroot)
            

                slice_im_cowc(imfile_tot, annotate_file_tot, outroot, out_test_im, 
                             out_test_label, classes_dic, category, yolt_box_size, 
                             sliceHeight=sliceHeight, sliceWidth=sliceWidth, 
                             zero_frac_thresh=zero_frac_thresh, 
                             overlap=slice_overlap, 
                             pad=0, verbose=False,
                             box_coords_dir=out_test_box_coords)     


###############################################################################
###############################################################################
if __name__ == "__main__":
    main()
import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json
import patch_config

    
def test(imgdir, model, patchfile, savedir, 
         conf_thresh=0.4, nms_thresh=0.4, 
         save_images=False, cls_id=None,
         max_ims=100000,
         verbose=False, super_verbose=False):
    
    print("Setting everything up")
    save_padded_image = False
    t0 = time.time()
    config = patch_config.patch_configs[model]()

    darknet_model = Darknet(config.cfgfile)
    darknet_model.load_weights(config.weightfile)
    darknet_model = darknet_model.eval().cuda()
    patch_applier = PatchApplier(patch_alpha=config.patch_alpha).cuda()
    patch_transformer = PatchTransformer(target_size_frac=config.target_size_frac).cuda()

    batch_size = config.batch_size
    max_lab = config.max_lab
    img_size = darknet_model.height
    patch_size = config.patch_size

    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size,patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()
    
    # make dirs
    # make dirs
    cleandir = os.path.join(savedir, 'clean/')
    txtdir = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/'))
    properdir = os.path.join(savedir, 'proper_patched/')
    txtdir2 = os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels/'))
    randomdir = os.path.join(savedir, 'random_patched/')
    txtdir3 = os.path.abspath(os.path.join(savedir, 'random_patched/', 'yolo-labels/'))
    jsondir = os.path.join(savedir, 'jsons')
    for z in (cleandir, txtdir, properdir, txtdir2, randomdir, txtdir3, jsondir):
        print("output dir:", z)
        os.makedirs(z, exist_ok=False)
            
    clean_results = []
    noise_results = []
    patch_results = []
    
    t1 = time.time()
    print("Done setting up, took {} seconds".format(t1-t0))
    list_tmp = os.listdir(imgdir)
    im_list = sorted([z for z in list_tmp if (z.endswith('.jpg') or z.endswith('.png'))])
    print("Total num images:", len(im_list))
    im_list = im_list[:max_ims]
    print("len im_list:", len(im_list))
    
    ##############
    # Clean images
    #Loop over cleane beelden
    for i, imgfile in enumerate(im_list):
        ti0 = time.time()
        # print("new image")
        # if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
        name = os.path.splitext(imgfile)[0]    #image name w/o extension
        if (i % 50) == 0:
            print (i, "/", len(im_list), name)
        if verbose:
            print (i, "/", len(im_list), name)            
        txtname = name + '.txt'
        txtpath = os.path.join(txtdir, txtname)
        # open beeld en pas aan naar yolo input size
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
        img = Image.open(imgfile).convert('RGB')
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
        resize = transforms.Resize((img_size,img_size))
        padded_img = resize(padded_img)
        cleanname = name + ".png"
        #sla dit beeld op
        if save_images and save_padded_image:
            padded_img.save(os.path.join(cleandir, cleanname))
        
        #genereer een label file voor het gepadde beeld
        boxes = do_detect(darknet_model, padded_img, conf_thresh, nms_thresh, True)
        boxes = nms(boxes, nms_thresh)
        textfile = open(txtpath,'w+')
        for box in boxes:
            cls_id_box = box[6]
            # if(cls_id == 0):   #if person
            if super_verbose:
                print("  cls_id_box", cls_id_box)
            x_center = box[0]
            y_center = box[1]
            width = box[2]
            height = box[3]
            textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
            clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                 y_center.item() - height.item() / 2,
                                                                 width.item(),
                                                                 height.item()],
                                      'score': box[4].item(),
                                      'category_id': cls_id_box.item()})
        textfile.close()
        ti1 = time.time()
        if verbose:
            print(" Time to compute clean results = {} seconds".format(ti1 - ti0))
            
        #############
        # Apply patch
        #lees deze labelfile terug in als tensor            
        if os.path.getsize(txtpath):       #check to see if label file contains data. 
            label = np.loadtxt(txtpath)
        else:
            label = np.ones([5])
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)
        
        transform = transforms.ToTensor()
        padded_img2 = transform(padded_img).cuda()
        img_fake_batch = padded_img2.unsqueeze(0)
        lab_fake_batch = label.unsqueeze(0).cuda()
        if verbose:
            print(" img_fake_batch.shape", img_fake_batch.shape)
            print(" lab_fake_batch.shape", lab_fake_batch.shape)
        
        # Optional, Filter lab_batch array to only include desired cls_id
        if cls_id != None:
            # transform to numpy so we can filter
            lab_squeeze = label.numpy()  #lab_fake_batch.squeeze(0).numpy()
            if verbose:
                print("  lab_squeeze.shape:", lab_squeeze.shape)
            # ztmp = lab_squeeze.numpy()
            # filter out undesired labels
            good_idxs = []
            for i, row_tmp in enumerate(lab_squeeze):
                if super_verbose:
                    print("  row:", row_tmp)
                # rows of [1., 1., 1., 1., 1.] are filler
                if np.array_equal(row_tmp, [1., 1., 1., 1., 1.]):
                    continue
                # if not the desired cls_id, skip
                elif int(row_tmp[0]) != config.cls_id:
                    continue
                else:
                    good_idxs.append(i)
            if verbose:
                print("  good_idxs:", good_idxs)
            if len(good_idxs) > 0:
                use_clean_boxes = False
                lab_squeeze_filt = lab_squeeze[good_idxs]
                lab_fake_batch = torch.from_numpy(lab_squeeze_filt).float().unsqueeze(0).cuda()
                if verbose:
                    print("  lab_batch_filt.shape", lab_squeeze_filt.shape)
                    print("  lab_fake_batch.shape", lab_fake_batch.shape)
            else:
                use_clean_boxes = True
                
        #transformeer patch en voeg hem toe aan beeld
        if not use_clean_boxes:
            adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
        else:
            p_img_pil = padded_img
        properpatchedname = name + "_p.png"
        if save_images: 
            p_img_pil.save(os.path.join(properdir, properpatchedname))
        
        #genereer een label file voor het beeld met sticker       
        txtname = properpatchedname.replace('.png', '.txt')
        txtpath = os.path.join(txtdir2, txtname)
        boxes = do_detect(darknet_model, p_img_pil, conf_thresh, nms_thresh, True)
        boxes = nms(boxes, nms_thresh)
        textfile = open(txtpath,'w+')
        for box in boxes:
            cls_id_box = box[6]
            # if(cls_id == 0):   #if person
            x_center = box[0]
            y_center = box[1]
            width = box[2]
            height = box[3]
            textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
            patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': cls_id_box.item()})
        textfile.close()
        ti2 = time.time()
        if verbose:
            print(" Time to compute proper patched results = {} seconds".format(ti2 - ti1))
        
        #maak een random patch, transformeer hem en voeg hem toe aan beeld
        random_patch = torch.rand(adv_patch_cpu.size()).cuda()
        adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
        p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        p_img = p_img_batch.squeeze(0)
        p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
        properpatchedname = name + "_rdp.png"
        if save_images:
            p_img_pil.save(os.path.join(randomdir, properpatchedname))
        
        #genereer een label file voor het beeld met random patch
        txtname = properpatchedname.replace('.png', '.txt')
        txtpath = os.path.join(txtdir3, txtname)
        boxes = do_detect(darknet_model, p_img_pil, conf_thresh, nms_thresh, True)
        boxes = nms(boxes, nms_thresh)
        textfile = open(txtpath,'w+')
        for box in boxes:
            cls_id_box = box[6]
            # if(cls_id == 0):   #if person
            x_center = box[0]
            y_center = box[1]
            width = box[2]
            height = box[3]
            textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
            noise_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2, y_center.item() - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': cls_id_box.item()})
        textfile.close()
        ti3 = time.time()
        if verbose:
            print(" Time to compute random results = {} seconds".format(ti3 - ti2))
            print(" Total time to compute results = {} seconds".format(ti3 - ti0))
    
    if super_verbose:
        print("clean_results:", clean_results)
    # save results
    with open(os.path.join(jsondir, 'clean_results.json'), 'w') as fp:
        json.dump(clean_results, fp)
    with open(os.path.join(jsondir, 'noise_results.json'), 'w') as fp:
        json.dump(noise_results, fp)
    with open(os.path.join(jsondir, 'patch_results.json'), 'w') as fp:
        json.dump(patch_results, fp)
        
    tf = time.time()
    print(" Time to compute proper patched results = {} seconds".format(tf - t0))     
    return
    
    
if __name__ == '__main__':
    
    if len(sys.argv) < 5:
        print('Usage:')
        print('python test_patch.py imgdir model patchfile savedir ...')
        sys.exit("Exiting. Fewer than 5 arguments...")
    
    imgdir = sys.argv[1]  # "inria/Test/pos"
    model = sys.argv[2]  # "cfg/yolo.cfg"
    patchfile = sys.argv[3]  # "saved_patches/patch11.jpg"
    savedir = sys.argv[4]  # "testing"
    
    test(imgdir, model, patchfile, savedir)

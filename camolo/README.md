# Adversarial YOLO
This repository is based on the marvis YOLOv2 inplementation: https://github.com/marvis/pytorch-yolo2
https://gitlab.com/EAVISE/adversarial-yolo
https://arxiv.org/abs/1904.08653


!!
Try editing line 255 of darknet.py to load as np.floa64 of header minor version is >=2???
The "seen" variable in the header of the .weights file is stored as int64 in more recent versions of yolo.  
If this is loaded as 32-bit (as is the default in adversarial-yolo) the weights get loaded incorrectly
and prediction/training yields nans.
E.g. Darknet  weight header: major 0, minor 2, revision 0, seen 1280000
If "seen" is not loaded as an int64 for minor revision >=2: the weights fail
!! 


----------
AVE NOTES

Data: https://gdo152.llnl.gov/cowc/

list of scipts with hardcoded paths or variables
	batch_detect.py - does not appear to be used in other scripts
	batch_rotate.py  - does not appear to be used in other scripts
	detect.py
	load_data.py
	patch_config.py - no need to edit because variables can be edited when this script is called
	test_patch.py
	train_pathc.py - uses InriaDataset
	


list of scipts edited by AVE
	batch_detect.py (add support for num_classes == 1)
	detect.py  (add support for num_classes == 1)
	load_data.py (add COWCDataset class)
	patch_config.py (no edits as of yet...)
	train_patch.py - replace InriaDataset with COWCDataset
	darknet.py - fix loading of weights
	
----------
ORIGINAL README BELOW HERE

This work corresponds to the following paper: https://arxiv.org/abs/1904.08653:
```
@inproceedings{thysvanranst2019,
    title={Fooling automated surveillance cameras: adversarial patches to attack person detection},
    author={Thys, Simen and Van Ranst, Wiebe and Goedem\'e, Toon},
    booktitle={CVPRW: Workshop on The Bright and Dark Sides of Computer Vision: Challenges and Opportunities for Privacy and Security},
    year={2019}
}
```

If you use this work, please cite this paper.

# What you need
We use Python 3.6.
Make sure that you have a working implementation of PyTorch installed, to do this see: https://pytorch.org/
To visualise progress we use tensorboardX which can be installed using pip:
```
pip install tensorboardX tensorboard
```
No installation is necessary, you can simply run the python code straight from this directory.

Make sure you have the YOLOv2 MS COCO weights:
```
mkdir weights; curl https://pjreddie.com/media/files/yolov2.weights -o weights/yolo.weights
```

Get the INRIA dataset:
```
curl ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar -o inria.tar
tar xf inria.tar
mv INRIAPerson inria
cp -r yolo-labels inria/Train/pos/
```

# Generating a patch
`patch_config.py` contains configuration of different experiments. You can design your own experiment by inheriting from the base `BaseConfig` class or an existing experiment. `ReproducePaperObj` reproduces the patch that minimizes object score from the paper (With a lower batch size to fit on a desktop GPU).

You can generate this patch by running:
```
python train_patch.py paper_obj
```

from torch import optim

'''
https://arxiv.org/pdf/1904.08653.pdf
Ltv The total variation in the image as described in [17]. This loss makes sure that our optimiser favours an image with smooth colour transitions and prevents noisy images. The score is low if neighbouring pixels are similar, and high if neighbouring pixel are different.
'''


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.src_dir = "/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt"
        self.img_dir = "inria/Train/pos"
        self.lab_dir = "inria/Train/pos/yolo-labels"
        self.cfgfile = "cfg/yolo.cfg"
        self.weightfile = "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patchfile = "gray" # can be 'gray', 'random', or path to init patch
        self.patch_size = 300
        self.num_cls = 80
        self.n_epochs = 1000
        self.max_lab = 14
        self.start_learning_rate = 0.03
        self.patch_name = 'base'
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0
        self.tv_mult = 2.5
        self.nps_mult = 0.01
        self.batch_size = 20
        self.loss_target = lambda obj, cls: obj * cls
        
#####################
## AVE Experiments ##
#####################

# class visdrone_v1_4cat_obj_only_gray_v2p4(BaseConfig):
#     """
#     VisDrone: Generate a patch that minimises object score.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#         self.class_name = self.__class__.__name__
#         self.patch_name = 'patch_yolt2_ave_26x26_alpha0p8_grays_' + self.class_name
#         self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
#         self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
#         self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
#         self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
#         self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
#         self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2_epoch35.jpg'
#         self.patch_size = 256
#         self.target_size_frac = 0.3
#         self.patch_alpha = 0.6
#         self.cls_id = 0
#         self.num_cls = 4
#         self.batch_size = 12
#         self.n_epochs = 1000
#         self.max_lab = 40
#         self.start_learning_rate = 0.05
#         self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
#         self.max_tv = 0.165
#         self.loss_target = lambda obj, cls: obj


class visdrone_v1_4cat_obj_only_tiny_gray_v1(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p75_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2_epoch38.jpg'
        # self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha1_visdrone_v1_4cat_obj_only_tiny_gray_v0/patch_yolt2_ave_26x26_alpha1_visdrone_v1_4cat_obj_only_tiny_gray_v0_epoch200.jpg' 
        self.patch_size = 16
        self.target_size_frac = 0.2
        self.patch_alpha = 0.75
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.tv_mult = 0.1
        self.nps_mult = 5
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.loss_target = lambda obj, cls: obj


class visdrone_v1_4cat_obj_only_tiny_gray_v0(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha1_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5_epoch66.jpg' 
        self.patch_size = 16
        self.target_size_frac = 0.25
        self.patch_alpha = 1.0
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.tv_mult = 0.25
        self.nps_mult = 4
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.loss_target = lambda obj, cls: obj
     

class visdrone_v1_4cat_obj_only_tiny_v0(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha1_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5_epoch66.jpg' 
        self.patch_size = 16
        self.target_size_frac = 0.25
        self.patch_alpha = 1.0
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.tv_mult = 0.25
        self.nps_mult = 0.005
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.loss_target = lambda obj, cls: obj
     
     
class visdrone_v1_4cat_class_only_v1(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p9_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2_epoch35.jpg' 
        self.patch_size = 128
        self.target_size_frac = 0.25
        self.patch_alpha = 0.9
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.loss_target = lambda obj, cls: cls
     

class visdrone_v1_4cat_obj_only_gray_v2p3(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p8_grays_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2_epoch35.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.2
        self.patch_alpha = 0.7
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj
        
class visdrone_v1_4cat_obj_only_gray_v2p2(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p8_grays_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2_epoch35.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.25
        self.patch_alpha = 0.6
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj
        
class visdrone_v1_4cat_obj_only_gray_v2p1(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p8_grays_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2/patch_yolt2_ave_26x26_alpha0p8_grays_visdrone_v1_4cat_obj_only_gray_v2_epoch35.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.25
        self.patch_alpha = 0.7
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj



class visdrone_v1_4cat_obj_only_v5p3(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p5_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5_epoch66.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.2
        self.patch_alpha = 0.4
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj

class visdrone_v1_4cat_obj_only_v5p2(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p5_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5_epoch66.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.25
        self.patch_alpha = 0.4
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj


class visdrone_v1_4cat_obj_only_v5p1(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p5_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5/patch_yolt2_ave_26x26_alpha0p5_32vals_visdrone_v1_4cat_obj_only_v5_epoch66.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.25
        self.patch_alpha = 0.5
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj


class visdrone_v1_4cat_obj_only_v5(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p5_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_30vals_obj_only_v0_epoch233.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.3
        self.patch_alpha = 0.5
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj


class visdrone_v1_4cat_obj_only_v4(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p5_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_30vals_obj_only_v0_epoch233.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.12
        self.patch_alpha = 0.5
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj
        
        
class visdrone_v1_4cat_obj_class_v4(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p5_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_30vals_obj_only_v0_epoch233.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.12
        self.patch_alpha = 0.5
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj * cls


class visdrone_v1_4cat_obj_class_v3(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p6_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_30vals_obj_only_v0_epoch233.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.16
        self.patch_alpha = 0.6
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj * cls
        

class visdrone_v1_4cat_obj_only_v2(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p8_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_30vals_obj_only_v0_epoch233.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.25
        self.patch_alpha = 0.8
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj


class visdrone_v1_4cat_obj_only_gray_v2(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p8_grays_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/data/patches/arxiv_2008.13671/small_gray_patch.png' 
        self.patch_size = 256
        self.target_size_frac = 0.25
        self.patch_alpha = 0.8
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj


class visdrone_v1_4cat_obj_class_small_v2(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p75_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/patch_yolt2_ave_26x26_alpha0p6_32vals_visdrone_v1_4cat_obj_only_small_v1/patch_yolt2_ave_26x26_alpha0p6_32vals_visdrone_v1_4cat_obj_only_small_v1_epoch100.jpg' 
        self.patch_size = 128
        self.target_size_frac = 0.16
        self.patch_alpha = 0.75
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=20)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj * cls


class visdrone_v1_4cat_obj_only_small_v1(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p6_32vals_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/data/patches/arxiv_2008.13671/small_patch.png' 
        self.patch_size = 128
        self.target_size_frac = 0.16
        self.patch_alpha = 0.6
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 8
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj
        
class visdrone_obj_only_small_gray_v1(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p3_30vals_obj_only_small_gray_v1'
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/data/patches/arxiv_2008.13671/small_gray_patch.png' 
        self.patch_size = 128
        self.target_size_frac = 0.16
        self.patch_alpha = 0.4
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 8
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj
        

class visdrone_obj_only_small_gray_v0(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p3_30vals_obj_only_small_gray_v0'
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v0.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/ave_legacy/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_32vals_small_epoch14.jpg' 
        self.patch_size = 128
        self.target_size_frac = 0.20
        self.patch_alpha = 0.3
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 8
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj
        
        
class visdrone_obj_only_small_v0(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p3_30vals_obj_only_small_v0'
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/30values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/ave_legacy/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_32vals_small_epoch14.jpg' 
        self.patch_size = 128
        self.target_size_frac = 0.20
        self.patch_alpha = 0.3
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 8
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj


class visdrone_obj_only_v0(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_30vals_obj_only_v0'
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile = '//local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/30values.txt'
        self.patchfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/saved_patches/ave_legacy/patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_32vals_epoch10.jpg' 
        self.patch_size = 256
        self.target_size_frac = 0.20
        self.patch_alpha = 0.4
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 8
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj


class visdrone_obj_class_v0(BaseConfig):
    """
    VisDrone: Generate a patch that minimises object and class score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_32vals_obj_class_v0'
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/30values.txt'
        self.patchfile = "gray" 
        self.patch_size = 256
        self.target_size_frac = 0.20
        self.patch_alpha = 0.4
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 8
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj * cls


class visdrone_class_only_v0(BaseConfig):
    """
    Generate a patch that minimises class score.
    """

    def __init__(self):
        super().__init__()
        
        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_visdrone_v1_4cat_alpha0p4_32vals_class_only_v0'
        self.cfgfile = '/local_data/cosmiq/src/avanetten/yoltv4/darknet/cfg/_backup/yolt2_ave_26x26_visdrone_v1_4cat.cfg'
        self.weightfile= '//local_data/cosmiq/src/avanetten/camouflage/weights/VisDrone/yolt2_ave_26x26_visdrone_v1_4cat_final.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/images_only'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/VisDrone/data/VisDrone2019-DET-train/yolt/v1_4cat/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/30values.txt'
        self.patchfile = "gray" 
        self.patch_size = 256
        self.target_size_frac = 0.20
        self.patch_alpha = 0.4
        self.cls_id = 0
        self.num_cls = 4
        self.batch_size = 8
        self.n_epochs = 1000
        self.max_lab = 40      
        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: cls

##################################

class COWC_obj_only_v4(BaseConfig):
    """
    COWC: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha1_grat' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26_55000_tmp.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/images'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_subtle_grays_v1.txt'
        self.patchfile = '/local_data/cosmiq/wdata/camouflage/data/patches/arxiv_2008.13671/small_gray_patch.png'
        self.patch_size = 16
        self.target_size_frac = 0.35
        self.patch_alpha = 1
        self.cls_id = 0        
        self.num_cls = 1
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 30 
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.tv_mult = 0.1
        self.nps_mult = 2
        self.loss_target = lambda obj, cls: obj
        
        
class COWC_obj_only_v3(BaseConfig):
    """
    COWC: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p15_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26_55000_tmp.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/images'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/wdata/camouflage/data/patches/arxiv_2008.13671/large_patch.png'
        self.patch_size = 24
        self.target_size_frac = 0.35
        self.patch_alpha = 0.15
        self.cls_id = 0        
        self.num_cls = 1
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 30 
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.tv_mult = 0.5
        self.nps_mult = 0.005
        self.loss_target = lambda obj, cls: obj
        

class COWC_obj_only_v2(BaseConfig):
    """
    COWC: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha0p3_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26_55000_tmp.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/images'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/wdata/camouflage/data/patches/arxiv_2008.13671/large_patch.png'
        self.patch_size = 30
        self.target_size_frac = 0.35
        self.patch_alpha = 0.3
        self.cls_id = 0        
        self.num_cls = 1
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 30 
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.tv_mult = 2.5
        self.nps_mult = 0.01
        self.loss_target = lambda obj, cls: obj
        

class COWC_obj_only_v1(BaseConfig):
    """
    COWC: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.class_name = self.__class__.__name__
        self.patch_name = 'patch_yolt2_ave_26x26_alpha1_' + self.class_name
        self.cfgfile = '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26_55000_tmp.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/images'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/cowc_32values.txt'
        self.patchfile = '/local_data/cosmiq/wdata/camouflage/data/patches/arxiv_2008.13671/large_patch.png'
        self.patch_size = 30
        self.target_size_frac = 0.35
        self.patch_alpha = 1
        self.cls_id = 0        
        self.num_cls = 1
        self.batch_size = 12
        self.n_epochs = 1000
        self.max_lab = 30 
        self.start_learning_rate = 0.05
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=10)
        self.tv_mult = 2.5
        self.nps_mult = 0.01
        self.loss_target = lambda obj, cls: obj


class COWC_obj_only(BaseConfig):
    """
    COWC: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.class_name = self.__class__.__name__
        self.cfgfile = '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26.cfg'
        self.weightfile= '/local_data/cosmiq/src/avanetten/camouflage/weights/cowc_15cm/ave_26x26_55000_tmp.weights'
        self.img_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/images'
        self.lab_dir = '/local_data/cosmiq/wdata/avanetten/cowc/wdata/camo_train/labels'
        self.printfile = '/local_data/cosmiq/src/avanetten/camouflage/adversarial-yolt/non_printability/30values.txt'
        self.patch_size = 300
        self.num_cls = 1
        self.n_epochs = 1000
        self.max_lab = 30 
        self.start_learning_rate = 0.03
        self.patch_name = 'base'
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.patch_name = 'COWC_Obj_Only'
        self.max_tv = 0.165
        self.loss_target = lambda obj, cls: obj
        

################
## Historical ##

class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400
        
        
class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 8
        self.patch_size = 300

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


# Config dict we will call in train_patch.py
patch_configs = {
    # Historical
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj,
    # AVE experiments
    "cowc_obj_only_v4": COWC_obj_only_v4,
    "cowc_obj_only_v3": COWC_obj_only_v3,
    "cowc_obj_only_v2": COWC_obj_only_v2,
    "cowc_obj_only_v1": COWC_obj_only_v1,
    "cowc_obj_only": COWC_obj_only,
    "visdrone_obj_only_v0": visdrone_obj_only_v0,
    "visdrone_obj_class_v0": visdrone_obj_class_v0,
    "visdrone_class_only_v0": visdrone_class_only_v0,
    "visdrone_obj_only_small_v0": visdrone_obj_only_small_v0,
    "visdrone_obj_only_small_gray_v0": visdrone_obj_only_small_gray_v0,
    "visdrone_obj_only_small_gray_v1": visdrone_obj_only_small_gray_v1,
    "visdrone_v1_4cat_obj_only_small_v1": visdrone_v1_4cat_obj_only_small_v1,
    "visdrone_v1_4cat_obj_class_small_v2": visdrone_v1_4cat_obj_class_small_v2,
    "visdrone_v1_4cat_obj_only_v2": visdrone_v1_4cat_obj_only_v2,
    "visdrone_v1_4cat_obj_only_gray_v2": visdrone_v1_4cat_obj_only_gray_v2,
    "visdrone_v1_4cat_obj_class_v3": visdrone_v1_4cat_obj_class_v3,
    "visdrone_v1_4cat_obj_class_v4": visdrone_v1_4cat_obj_class_v4,
    "visdrone_v1_4cat_obj_only_v4": visdrone_v1_4cat_obj_only_v4,
    "visdrone_v1_4cat_obj_only_v5": visdrone_v1_4cat_obj_only_v5,
    "visdrone_v1_4cat_obj_only_v5p1": visdrone_v1_4cat_obj_only_v5p1,
    "visdrone_v1_4cat_obj_only_v5p2": visdrone_v1_4cat_obj_only_v5p2,
    "visdrone_v1_4cat_obj_only_v5p3": visdrone_v1_4cat_obj_only_v5p3,
    "visdrone_v1_4cat_obj_only_gray_v2p1": visdrone_v1_4cat_obj_only_gray_v2p1,
    "visdrone_v1_4cat_obj_only_gray_v2p2": visdrone_v1_4cat_obj_only_gray_v2p2,
    "visdrone_v1_4cat_obj_only_gray_v2p3": visdrone_v1_4cat_obj_only_gray_v2p3,
    # "visdrone_v1_4cat_obj_only_gray_v2p4": visdrone_v1_4cat_obj_only_gray_v2p4
    "visdrone_v1_4cat_class_only_v1": visdrone_v1_4cat_class_only_v1,
    "visdrone_v1_4cat_obj_only_tiny_v0": visdrone_v1_4cat_obj_only_tiny_v0,
    "visdrone_v1_4cat_obj_only_tiny_gray_v0": visdrone_v1_4cat_obj_only_tiny_gray_v0,
    "visdrone_v1_4cat_obj_only_tiny_gray_v1": visdrone_v1_4cat_obj_only_tiny_gray_v1
}
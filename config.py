import os
from src.dataset.cityscapes_laf_mixed import CityscapesLAFMix
from src.dataset.laf_ood_proxy_predict import LostAndFound
from src.dataset.cityscapes import Cityscapes
from src.dataset.small_scale_obstacles import SmallScale

TRAINSETS   = ["Cityscapes+LAF"]
VALSETS     = ["LostAndFound", "SmallScale"]
MODELS      = ["DeepLabV3+_WideResNet38","DeepLabV3+_ResNet101"]

TRAINSET    = TRAINSETS[0]
VALSET      = VALSETS[0]
MODEL       = MODELS[0]
IO          = "/home/ravi/ravivaghasiya/cs_laf_predict"
class cs_laf_roots:
    
    """
    OoD training roots for Cityscapes + LAF mix
    """
    def __init__(self,args) :
        self.model_name = MODELS[0] if args['backbone'] == 'WideResNet38' else MODELS[1]
        self.init_ckpt   = os.path.join("/home/ravi/ravivaghasiya/dataset/cityscapes/weights", self.model_name + ".pth")
        self.cs_root     = "/home/ravi/ravivaghasiya/dataset/cityscapes"
        self.laf_root   = "/home/ravi/ravivaghasiya/dataset/LostAndFound"
        self.io_root     = os.path.join(IO,"_ood_disp_" + self.model_name)
        self.weights_dir = os.path.join(self.io_root, "weights/")

class laf_roots:
    """
    LostAndFound config class
    """
    def __init__(self,args) -> None:
        self.model_name = MODELS[0] if args['backbone'] == 'WideResNet38' else MODELS[1]
        self.init_ckpt = os.path.join("/home/ravi/ravivaghasiya/dataset/cityscapes/weights", self.model_name + ".pth")
        self.eval_dataset_root = "/home/ravi/ravivaghasiya/dataset/LostAndFound"
        self.eval_sub_dir = "laf_eval"
        self.io_root = os.path.join(IO,"_ood_disp_" + self.model_name, self.eval_sub_dir)
        self.weights_dir = os.path.join(IO,"_ood_disp_" + self.model_name, "weights/")
    
class cs_roots:
    """
    Cityscapes config class
    """
    def __init__(self,args) -> None:
        self.model_name = MODELS[0] if args['backbone'] == 'WideResNet38' else MODELS[1]
        self.cs_root     = "/home/ravi/ravivaghasiya/dataset/cityscapes"
        self.init_ckpt = os.path.join("/home/ravi/ravivaghasiya/dataset/cityscapes/weights", self.model_name + ".pth")
        self.eval_dataset_root = "/home/ravi/ravivaghasiya/dataset/cityscapes"
        self.eval_sub_dir = "cs_eval"
        self.io_root = os.path.join(IO,"_ood_disp_" + self.model_name, self.eval_sub_dir)
        self.weights_dir = os.path.join(IO,"_ood_disp_" + self.model_name, "weights/")

class small_obstacles_roots:
    """
    Cityscapes config class
    """
    def __init__(self,args) -> None:
        self.model_name = MODELS[0] if args['backbone'] == 'WideResNet38' else MODELS[1]
        self.cs_root     = "/home/ravi/ravivaghasiya/dataset/Small_obstacles"
        self.init_ckpt = os.path.join("/home/ravi/ravivaghasiya/dataset/cityscapes/weights", self.model_name + ".pth")
        self.eval_dataset_root = "/home/ravi/ravivaghasiya/dataset/Small_obstacles"
        self.eval_sub_dir = "small_obstacles_eval"
        self.io_root = os.path.join(IO,"_ood_disp_" + self.model_name, self.eval_sub_dir)
        self.weights_dir = os.path.join(IO,"_ood_disp_" + self.model_name, "weights/")
    
class params:
    """
    Set pipeline parameters
    """
    training_starting_epoch = 81
    num_training_epochs     = 115
    pareto_alpha            = 0.9
    ood_subsampling_factor  = 0.1
    learning_rate           = 1e-4#1e-5
    crop_size               = 480
    val_epoch               = num_training_epochs
    batch_size              = 8
    entropy_threshold       = 0.7


#########################################################################

class config_training_setup(object):
    """
    Setup config class for training
    If 'None' arguments are passed, the settings from above are applied
    """
    def __init__(self, args):
        if args['weighting_method'] == 'dwa' :
            self.WEIGHTING_METHOD = 'dwa'
        elif  args['weighting_method'] == 'uw' :
            self.WEIGHTING_METHOD = 'uw'
        elif  args['weighting_method'] == 'adamtnet' :
            self.WEIGHTING_METHOD = 'adamtnet'
        else: 
            self.WEIGHTING_METHOD = 'Fix'
        
            
        if args['architecture'] == 'shared_decoder':
            self.architecture = 'shared_decoder'
        elif args['architecture'] == 'sep_decoder':
            self.architecture = 'sep_decoder'
        elif args['architecture'] == 'osdnet_v3':
            self.architecture = 'osdnet_v3'
        elif args['architecture'] == 'osdnet_v4':
            self.architecture = 'osdnet_v4'
        elif args['architecture'] == 'osdnet_v5':
            self.architecture = 'osdnet_v5'
        else:
            print ('please specify correct architecture')
            exit()
        backbones = ['WideResNet38','ResNet101']
        if args['backbone'] in backbones:
            self.backbone = args['backbone']
        else:
            print('Please Select either WideResNet38 or ResNet101 as backbone')
            exit()
        if args["TRAINSET"] is not None:
            self.TRAINSET = args["TRAINSET"]
        else:
            self.TRAINSET = TRAINSET
       
        if self.TRAINSET == "Cityscapes+LAF":
            self.roots=cs_laf_roots(args)
            self.dataset=CityscapesLAFMix

        else:
            print("TRAINSET not correctly specified... bye...")
            exit()

        if args["MODEL"] is not None:
            tmp = getattr(self.roots, "model_name")
            roots_attr = [attr for attr in dir(self.roots) if not attr.startswith('__')]
            for attr in roots_attr:
                if tmp in getattr(self.roots, attr):
                    rep = getattr(self.roots, attr).replace(tmp, args["MODEL"])
                    setattr(self.roots, attr, rep)
        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith('__')]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.weights_dir]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)

class config_evaluation_setup(object):
    """
    Setup config class for evaluation
    If 'None' arguments are passed, the settings from above are applied
    """
    def __init__(self, args):
        if args["VALSET"] is not None:
            self.VALSET = args["VALSET"]
        else:
            self.VALSET = VALSET
        if self.VALSET == "LostAndFound":
            self.roots = laf_roots(args)
            self.dataset = LostAndFound
            self.split = 'test'
        if self.VALSET == "Cityscapes":
            self.roots = cs_roots(args)
            self.dataset = Cityscapes
            self.split = 'val'
        if self.VALSET == 'SmallScale':
            self.roots = small_obstacles_roots(args)
            self.dataset = SmallScale
            self.split = 'test'

        if args['architecture'] == 'shared_decoder':
            self.architecture = 'shared_decoder'
        elif args['architecture'] == 'sep_decoder':
            self.architecture = 'sep_decoder'
        elif args['architecture'] == 'osdnet_v3':
            self.architecture = 'osdnet_v3'
        elif args['architecture'] == 'osdnet_v4':
            self.architecture = 'osdnet_v4'
        elif args['architecture'] == 'osdnet_v5':
            self.architecture = 'osdnet_v5'
        else:
            print ('please specify correct architecture')
            exit()

        self.params = params
        params_attr = [attr for attr in dir(self.params) if not attr.startswith('__')]
        for attr in params_attr:
            if attr in args:
                if args[attr] is not None:
                    setattr(self.params, attr, args[attr])
        roots_attr = [self.roots.io_root]
        for attr in roots_attr:
            if not os.path.exists(attr):
                print("Create directory:", attr)
                os.makedirs(attr)
import os
from src.dataset.cityscapes_laf_mixed_ood_seg import CityscapesLAFMix
from src.dataset.laf_ood_proxy_predict_ood_seg import LostAndFound
from src.dataset.cityscapes_ood_seg import Cityscapes


TRAINSETS   = ["Cityscapes+LAF"]
VALSETS     = ["LostAndFound", "Fishyscapes"]
MODELS      = ["DeepLabV3+_WideResNet38", "DualGCNNet_res50"]

TRAINSET    = TRAINSETS[0]
VALSET      = VALSETS[0]
MODEL       = MODELS[0]
IO          = "/home/ravi/ravivaghasiya/cs_laf_predict"

class cs_laf_roots:
    """
    OoD training roots for Cityscapes + LAF mix
    """
    model_name  = MODEL
    init_ckpt   = os.path.join("/home/ravi/ravivaghasiya/dataset/cityscapes/weights", model_name + ".pth")
    cs_root     = "/home/ravi/ravivaghasiya/dataset/cityscapes"
    laf_root   = "/home/ravi/ravivaghasiya/dataset/LostAndFound"

    io_root     = os.path.join(IO,"_ood_segm_" + model_name)
    weights_dir = os.path.join(io_root, "weights/")

class laf_roots:
    """
    LostAndFound config class
    """
    model_name = MODEL
    init_ckpt = os.path.join("/home/ravi/ravivaghasiya/dataset/cityscapes/weights", model_name + ".pth")
    eval_dataset_root = "/home/ravi/ravivaghasiya/dataset/LostAndFound"
    eval_sub_dir = "laf_eval"
    io_root     = os.path.join(IO,"_ood_segm_" + model_name,eval_sub_dir)
    weights_dir = os.path.join(IO,"_ood_segm_" + model_name, "weights/")

class cs_roots:
    """
    Cityscapes config class
    """
    model_name = MODEL
    cs_root     = "/home/ravi/ravivaghasiya/dataset/cityscapes"
    init_ckpt = os.path.join("/home/ravi/ravivaghasiya/dataset/cityscapes/weights", model_name + ".pth")
    eval_dataset_root = "/home/ravi/ravivaghasiya/dataset/cityscapes"
    eval_sub_dir = "cs_eval"
    io_root     = os.path.join(IO,"_ood_segm_" + model_name,eval_sub_dir)
    weights_dir = os.path.join(IO,"_ood_segm_" + model_name, "weights/")


class params:
    """
    Set pipeline parameters
    """
    training_starting_epoch = 0
    num_training_epochs     = 99
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
        if args["TRAINSET"] is not None:
            self.TRAINSET = args["TRAINSET"]
        else:
            self.TRAINSET = TRAINSET
        if self.TRAINSET == "Cityscapes+LAF":
            self.roots=cs_laf_roots
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
            self.roots = laf_roots
            self.dataset = LostAndFound
        if self.VALSET == "Cityscapes":
            self.roots = cs_roots
            self.dataset = Cityscapes
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
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds,train=True):
        if train:
            #label_preds=label_preds.detach().max(dim=0)[1].cpu().numpy()
            label_preds=label_preds.detach().cpu().numpy()
            label_preds=np.argmax(label_preds,axis=0)
            label_trues = label_trues.detach().cpu().numpy()
            for lt, lp in zip(label_trues, label_preds):
                self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
            #self.confusion_matrix += self._fast_hist( label_trues.flatten(), label_preds.flatten() )
        else:
            label_preds=np.argmax(label_preds,axis=0)
            self.confusion_matrix += self._fast_hist( label_trues.flatten(), label_preds.flatten() )
            # for lt, lp in zip(label_trues, label_preds):
            #     self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        #freq = hist.sum(axis=1) / hist.sum()
        #fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]

class RMSE(object):
    """Root Mean Squared Error computational block for depth estimation.
    Args:
      ignore_val (float): value to ignore in the target
                          when computing the metric.
    Attributes:
      name (str): descriptor of the estimator.
    """

    def __init__(self, ignore_val=0):
        self.ignore_val = ignore_val
        self.name = "rmse"
        self.reset()

    def reset(self):
        self.num = 0.0
        self.den = 0.0

    def update(self, pred, gt,train=True):
        if train:
            pred=pred.squeeze()
            pred=pred*255
            pred=pred.detach().cpu().numpy()
            gt=gt.cpu().numpy()
        else:
            pred=pred.squeeze()
            pred=pred*255
        assert (
            pred.shape == gt.shape
        ), "Prediction tensor must have the same shape as ground truth"
        
        pred = np.abs(pred)
        idx = gt != self.ignore_val
        diff = (pred - gt)[idx]
        self.num += np.sum(diff ** 2)
        self.den += np.sum(idx)

    def val(self):
        return np.sqrt(self.num / self.den)


class AEPE(object):
    """Averaged End Point Error computational block for depth estimation.
    Args:
      ignore_val (float): value to ignore in the target
                          when computing the metric.
    Attributes:
      name (str): descriptor of the estimator.
    """

    def __init__(self, ignore_val=0):
        self.ignore_val = ignore_val
        self.name = "aepe"
        self.reset()

    def reset(self):
        self.num = 0.0
        self.den = 0.0
        self.mean = []
        
    def update(self, pred, gt,train=True):
        if train:
            pred=pred.squeeze()
            pred=pred*255
            pred=pred.detach().cpu().numpy()
            gt=gt.cpu().numpy()
        else:
            pred=pred.squeeze()
            pred=pred*255
        assert (
            pred.shape == gt.shape
        ), "Prediction tensor must have the same shape as ground truth"
        
        pred = np.abs(pred)
        idx = gt != self.ignore_val
        diff = np.abs((pred - gt)[idx])
        self.num += np.sum(diff)
        self.den += np.sum(idx)
        self.mean.append( self.num / self.den )

    def val(self):
        return  np.mean(self.mean)
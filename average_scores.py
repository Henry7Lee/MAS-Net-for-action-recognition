import numpy as np
import torch


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = output.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

save_path = './experiments/test/EN/'


#V1_8：1clip_1crop #top1:49.21020656136088 top5:77.89446276688075
# files_scores = [save_path + 'somethingv1_segment8_crops1_clips1_shape224.npz',
#                 save_path + 'somethingv1_segment8_crops1_clips1_shape224.npz'
#                 ]

#V1_8：2clips_3crops #top1:51.05016490192675 top5:79.46537059538275
# files_scores = [save_path + 'somethingv1_segment8_crops3_clips2_shape256.npz',
#                 save_path + 'somethingv1_segment8_crops3_clips2_shape256.npz'
#                 ]

#V1_16：1clip_1crop #top1:51.909390730775904 top5:80.22912688769311
# files_scores = [save_path + 'somethingv1_segment16_crops1_clips1_shape224.npz',
#                 save_path + 'somethingv1_segment16_crops1_clips1_shape224.npz'
#                 ]

#V1_16：2clips_3crops #top1:53.39350807151536 top5:81.77399756986634
# files_scores = [save_path + 'somethingv1_segment16_crops3_clips2_shape256.npz',
#                 save_path + 'somethingv1_segment16_crops3_clips2_shape256.npz'
#                 ]

#V1_8+1：1clip_1crop #top1:54.513105363652144 top5:82.19927095990279
# files_scores = [save_path + 'somethingv1_segment8_crops1_clips1_shape224.npz',
#                 save_path + 'somethingv1_segment16_crops1_clips1_shape224.npz'
#                 ]

#V1_8+16：2clips_3crops #top1:55.43308453393508 top5:83.2581149106058
files_scores = [save_path + 'somethingv1_segment8_crops3_clips2_shape256.npz',
                save_path + 'somethingv1_segment16_crops3_clips2_shape256.npz'
                ]
'''
#V2
###############################################################

#V2_8：1clip_1crop #top1:61.407757194172014 top5:87.03232836905194
# files_scores = [save_path + 'somethingv2_segment8_crops1_clips1_shape224.npz',
#                 save_path + 'somethingv2_segment8_crops1_clips1_shape224.npz'
#                 ]

#V2_8：2clips_3crops #top1:63.88182588691125 top5:88.73552084594584
# files_scores = [save_path + 'somethingv2_segment8_crops3_clips2_shape256.npz',
#                 save_path + 'somethingv2_segment8_crops3_clips2_shape256.npz'
#                 ]

#V2_16：1clip_1crop #top1:62.993905638293576 top5:88.47318077249062
# files_scores = [save_path + 'somethingv2_segment16_crops1_clips1_shape224.npz',
#                 save_path + 'somethingv2_segment16_crops1_clips1_shape224.npz'
#                 ]

#V2_16：2clips_3crops #top1:65.08859022480526 top5:89.84138515558784
# files_scores = [save_path + 'somethingv2_segment16_crops3_clips2_shape256.npz',
#                 save_path + 'somethingv2_segment16_crops3_clips2_shape256.npz'
#                 ]

#V2_8+16：1clip_1crop #top1:65.11280623158575 top5:89.5427210719619
# files_scores = [save_path + 'somethingv2_segment8_crops1_clips1_shape224.npz',
#                 save_path + 'somethingv2_segment16_crops1_clips1_shape224.npz'
#                 ]

#V2_8+16：2clips_3crops #top1:66.68684667231707 top5:90.65665738386407
# files_scores = [save_path + 'somethingv2_segment8_crops3_clips2_shape256.npz',
#                 save_path + 'somethingv2_segment16_crops3_clips2_shape256.npz'
#                 ]
'''

top1 = AverageMeter()
top5 = AverageMeter()


def compute_acc(labels, scores):
    preds_max = np.argmax(scores, 2)[:, 0]
    num_correct = np.sum(preds_max == labels)
    acc = num_correct * 1.0 / preds.shape[0]
    return acc

scores_agg = None
for filename in files_scores:
    data = np.load(filename)
    preds = data['predictions']
    labels = data['labels']
    scores = data['scores']
    if scores_agg is None:
        scores_agg = scores
    else:
        scores_agg += scores
    acc_scores = compute_acc(labels, scores)
    num_correct = np.sum(preds == labels)
    acc = num_correct * 1.0 / preds.shape[0]
for k in range(labels.shape[0]):
    label = torch.from_numpy(np.array([labels[k]]))
    score = torch.from_numpy(scores_agg[k,:])
    prec1, prec5 = accuracy(score, label, topk=(1, 5))
    top1.update(prec1.item(), 1)
    top5.update(prec5.item(), 1)
print('Accuracy of ensemble: top1:{} top5:{}'.format(top1.avg, top5.avg))

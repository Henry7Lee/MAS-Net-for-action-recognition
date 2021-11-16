import argparse
import time
from datetime import datetime

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet


from ops.models import TemporalModel
from ops.transforms import *
from ops import datasets_video
from torch.nn import functional as F
import os

def main():
    # options
    parser = argparse.ArgumentParser(description="MASNet testing on the full validation set")
    parser.add_argument('--dataset', type=str,default ='somethingv1') #somethingv1 #somethingv2 #kinetics
    parser.add_argument('--dataset_path', type = str, default = '../')

    # may contain splits
    #parser.add_argument('--weights', type=str, default="./experiments/test/v1_8/MASNet_somethingv1_resnet50_segment8_checkpoint.best.pth.tar")

    parser.add_argument('--weights', type=str, default="./experiments/test/v1_16/MASNet_somethingv1_resnet50_segment16_checkpoint.best.pth.tar")

    #parser.add_argument('--weights', type=str, default="./experiments/test/v2_8/MASNet_somethingv2_resnet50_segment8_checkpoint.best.pth.tar")
    #parser.add_argument('--weights', type=str, default="./experiments/test/v2_16/MASNet_somethingv2_resnet50_segment16_checkpoint.best.pth.tar")

    parser.add_argument('--test_segments', type=int, default=16)

    parser.add_argument('--dense_sample_num', type=int, default=10)
    parser.add_argument('--dense_sample', default=True, action="store_true", help='use dense sample as I3D')

    parser.add_argument('--twice_sample', default=True, action="store_true", help='use twice sample for ensemble')
    parser.add_argument('--test_crops', type=int, default=3)
    parser.add_argument('--full_res', default=True, action="store_true",
                        help='use full resolution 256x256 for test as in Non-local I3D')

    parser.add_argument('--coeff', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    # for true test
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--csv_file', type=str, default=None)

    parser.add_argument('--log_file', type=str, default="./experiments/test/test_log.txt")
    parser.add_argument('--save_scores', default=True, action="store_true")

    parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0,1,2,3])

    args = parser.parse_args()


    if args.dense_sample is True and args.twice_sample is False:
        clips = args.dense_sample_num
    elif args.twice_sample is True and args.dense_sample is False:
        clips = 2
    elif args.dense_sample is True and args.twice_sample is True:
        clips = 12
    else:
        clips = 1

    if args.full_res is True:
        input_size_log = 256
    else:
        input_size_log = 224

    log_input = ('dataset: {}, \tbatch_size: {},\ntest_segments: {}*{}*{},\tinput_shape: {},\nweights: " {} "\n\n'.format(
            args.dataset, args.batch_size, args.test_segments, args.test_crops, clips, input_size_log,
            args.weights))
    print(log_input)

    class AverageMeter(object):
        """Computes and stores the average and current value"""

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

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0)  # torch1.7
                # correct_k = correct[:k].view(-1).float().sum(0,keepdim = True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


    weights_list = [args.weights]
    test_segments_list = [int(args.test_segments)]

    assert len(weights_list) == len(test_segments_list)
    if args.coeff is None:
        coeff_list = [1] * len(weights_list)
    else:
        coeff_list = [float(c) for c in args.coeff.split(',')]

    if args.test_list is not None:
        test_file_list = args.test_list.split(',')
    else:
        test_file_list = [None] * len(weights_list)

    data_iter_list = []
    net_list = []
    modality_list = []

    total_num = None
    for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
        this_arch = "resnet50"
        modality = "RGB"
        new_length = 1
        modality_list.append(modality)


        categories, args.train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path)
        num_class = len(categories)


        net = TemporalModel(num_class, this_test_segments, backbone=this_arch)


        input_size = net.scale_size if args.full_res else net.input_size



        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose([
                GroupScale(net.scale_size),
                GroupCenterCrop(input_size),
            ])
        elif args.test_crops == 3:  # do not flip, so only 5 crops
            cropping = torchvision.transforms.Compose([
                GroupFullResSample(input_size, net.scale_size, flip=False)
            ])
        elif args.test_crops == 5:  # do not flip, so only 5 crops
            cropping = torchvision.transforms.Compose([
                GroupOverSample(input_size, net.scale_size, flip=False)
            ])
        elif args.test_crops == 10:
            cropping = torchvision.transforms.Compose([
                GroupOverSample(input_size, net.scale_size)
            ])
        else:
            raise ValueError("Only 1,3, 5, 10 crops are supported while we got {}".format(args.test_crops))



        data_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.dataset, root_path, test_file if test_file is not None else val_list,
                       num_segments=this_test_segments,
                       new_length=new_length,
                       modality=modality,
                       image_tmpl=prefix,
                       test_mode=True,
                       remove_missing=len(weights_list) == 1,
                       transform=torchvision.transforms.Compose([
                           cropping,
                           Stack(roll=False),
                           ToTorchFormatTensor(div=True),
                           GroupNormalize(net.input_mean, net.input_std),
                       ]), dense_sample=args.dense_sample,
                       dense_sample_num=args.dense_sample_num,
                       twice_sample=args.twice_sample),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        #net = torch.nn.DataParallel(net.cuda())
        if torch.cuda.is_available():
            net = torch.nn.DataParallel(net, device_ids=args.gpus).cuda()

        checkpoint = torch.load(this_weights)
        e_ = checkpoint['epoch']
        print("=> loaded checkpoint, (epoch {})".format(e_))
        net.load_state_dict(checkpoint['state_dict'])

        #     net = torch.nn.DataParallel(net.cuda())
        net.eval()

        data_gen = enumerate(data_loader)
        if total_num is None:
            total_num = len(data_loader.dataset)
        else:
            assert total_num == len(data_loader.dataset)

        data_iter_list.append(data_gen)
        net_list.append(net)

    output = []
    output_scores = []

    def eval_video(video_data, net, this_test_segments, modality):
        net.eval()
        with torch.no_grad():
            i, data, label = video_data
            batch_size = label.numel()
            num_crop = args.test_crops

            if args.twice_sample:
                num_crop *= 2

            if args.dense_sample:
                num_crop *= args.dense_sample_num  # 10 clips for testing when using dense sample


            if modality == 'RGB' :
                length = 3
            elif modality == 'Flow':
                length = 10
            elif modality == 'RGBDiff':
                length = 18
            else:
                raise ValueError("Unknown modality " + modality)
            # nt3, c, h, w
            data_in = data.view(-1, length*this_test_segments, data.size(-2), data.size(-1))
            rst = net(data_in)
            rst = rst.reshape(batch_size, num_crop, -1).mean(1)  # n, num_classes

            if args.softmax:
                # take the softmax to normalize the output to probability
                rst = F.softmax(rst, dim=1)

            rst = rst.data.cpu().numpy().copy()

            #         rst = rst.reshape(batch_size, num_class)

            return i, rst, label



    proc_start_time = time.time()
    max_num = args.max_num if args.max_num > 0 else total_num

    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, data_label_pairs in enumerate(zip(*data_iter_list)):
        with torch.no_grad():
            if i >= max_num:
                break
            this_rst_list = []
            this_label = None
            for n_seg, (_, (data, label)), net, modality in zip(test_segments_list, data_label_pairs, net_list,
                                                                modality_list):
                rst = eval_video((i, data, label), net, n_seg, modality)
                this_rst_list.append(rst[1])
                this_label = label

            assert len(this_rst_list) == len(coeff_list)
            for i_coeff in range(len(this_rst_list)):
                this_rst_list[i_coeff] *= coeff_list[i_coeff]
            ensembled_predict = sum(this_rst_list) / len(this_rst_list)

            for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
                output.append([p[None, ...], g])
            cnt_time = time.time() - proc_start_time

            rst_unsqueeze = np.expand_dims(rst[1],1)
            for j in range(int(rst_unsqueeze.shape[0])):
                rst_= np.expand_dims(rst_unsqueeze[j],axis=0)
                output_scores.append(np.mean(rst_, axis=0))

            prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
            top1.update(prec1.item(), this_label.numel())
            top5.update(prec5.item(), this_label.numel())
            if i % 20 == 0:
                print('video {} done, total {}/{}, time {:.3f} sec/video, '
                      'Prec@1  {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
                                                                  float(cnt_time) / (i + 1) / args.batch_size, top1.avg,
                                                                  top5.avg))
    video_pred = [np.argmax(x[0]) for x in output]
    video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

    video_labels = [x[1] for x in output]

    if args.csv_file is not None:
        print('=> Writing result to csv file: {}'.format(args.csv_file))
        with open(test_file_list[0].replace('test_videofolder.txt', 'category.txt')) as f:
            categories = f.readlines()
        categories = [f.strip() for f in categories]
        with open(test_file_list[0]) as f:
            vid_names = f.readlines()
        vid_names = [n.split(' ')[0] for n in vid_names]
        assert len(vid_names) == len(video_pred)
        if args.dataset != 'somethingv2':  # only output top1
            with open(args.csv_file, 'w') as f:
                for n, pred in zip(vid_names, video_pred):
                    f.write('{};{}\n'.format(n, categories[pred]))
        else:
            with open(args.csv_file, 'w') as f:
                for n, pred5 in zip(vid_names, video_pred_top5):
                    fill = [n]
                    for p in list(pred5):
                        fill.append(p)
                    f.write('{};{};{};{};{};{}\n'.format(*fill))

    cf = confusion_matrix(video_labels, video_pred).astype(float)

    # np.save('cm.npy', cf)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt
    # print(cls_acc)
    # upper = np.mean(np.max(cf, axis=1) / cls_cnt)
    # print('upper bound: {}'.format(upper))

    print('-----Evaluation is finished------')
    print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))





    if args.log_file is not None:
        log_test = open(args.log_file, 'a')
        log_test.write("********************************\n")
        log_test.write("[ " + str(datetime.now()).split(".")[0] + " ]\n")  # 打印时间

        log_test.write(log_input)

        log_test.write('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))
        log_test.write("\n********************************\n\n")

        if args.save_scores:

            # order_dict = {e:i for i, e in enumerate(sorted(name_list))}
            reorder_output = [None] * len(output)
            reorder_label = [None] * len(output)
            reorder_pred = [None] * len(output)
            output_csv = []


            for i in range(len(output)):
                reorder_output[i] = output_scores[i]
                reorder_label[i] = video_labels[i]
                reorder_pred[i] = video_pred[i]
            save_path = "./experiments/test/EN/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_name = args.dataset+'_segment'+str(args.test_segments)+'_crops'+str(args.test_crops) + '_clips' + str(clips) + '_shape' + str(input_size_log) + '.npz'
            np.savez(save_path+save_name, scores=reorder_output, labels=reorder_label, predictions=reorder_pred, cf=cf)


if __name__ == '__main__':
    main()

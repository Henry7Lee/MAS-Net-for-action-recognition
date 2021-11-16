import os
import time
import shutil
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from tensorboardX import SummaryWriter

from ops.dataset import TSNDataSet

from ops.models import TemporalModel
from ops.transforms import *
from opts import parser
from ops import datasets_video
from datetime import datetime

best_prec1 = 0

def main():
	global args, best_prec1
	args = parser.parse_args()

	# create folder
	check_rootfolders()

	# store_name
	global store_name,checkpoint_dir
	checkpoint_dir = os.path.join("experiments", args.dataset, args.type, args.arch, "num_segments_" + str(args.num_segments), args.store_name)
	store_name = '_'.join([args.type, args.dataset, args.arch, 'segment%d'% args.num_segments,args.store_name])
	print(('storing name: ' + store_name))

	# log_name
	log_training = open(os.path.join(checkpoint_dir, args.root_log, '%s.txt'%store_name), 'a')
	log_training.write("[ "+str(datetime.now()).split(".")[0]+" ]\n")  #打印时间

	# Save hyper parameters
	with open(os.path.join(checkpoint_dir, args.root_log, 'args.txt'), 'w') as f:
		f.write(str(args))

	# tensorboardX
	tf_writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tensorboardX"))

	# get dataset list
	categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset, args.dataset_path)
	num_class = len(categories)

	# create MASNet
	model = TemporalModel(num_class, num_segments=args.num_segments, base_model = args.type, backbone=args.arch,
						dropout = args.dropout)

	crop_size = model.crop_size
	scale_size = model.scale_size
	input_mean = model.input_mean
	input_std = model.input_std
	policies = model.get_optim_policies()

	# Something-something Dataset part label reverse
	train_augmentation = model.get_augmentation(
		flip=False if 'something' in args.dataset else True)

	# print params
	for group in policies:
		print(('[MASNet-{}] group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
			args.arch,group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
		log_training.write(('[MASNet-{}] group: {} has {} params, lr_mult: {}, decay_mult: {}\n'.format(
			args.arch,group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))


	# gpus
	if torch.cuda.is_available():
		model = nn.DataParallel(model,device_ids=args.gpus).cuda()

	# resume
	if args.resume:
		if os.path.isfile(args.resume):
			log_training.write("=> loading checkpoint '{}'\n".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print(("=> loaded checkpoint '{}' (epoch {})"
				   .format(args.evaluate, checkpoint['epoch'])))
			log_training.write("=> loaded checkpoint is evaluate'{}' (epoch {})\n"
					.format(args.evaluate, checkpoint['epoch']))
		else:
			log_training.write("=> no checkpoint found at '{}'\n".format(args.resume))
			print(("=> no checkpoint found at '{}'".format(args.resume)))

	if args.tune_from:
		print(("=> fine-tuning from '{}'".format(args.tune_from)))
		sd = torch.load(args.tune_from, "cpu")
		sd = sd['state_dict']
		model_dict = model.state_dict()
		replace_dict = []
		for k, v in sd.items():
			if k not in model_dict and k.replace('.net', '') in model_dict:
				print('=> Load after remove .net: ', k)
				replace_dict.append((k, k.replace('.net', '')))
		for k, v in model_dict.items():
			if k not in sd and k.replace('.net', '') in sd:
				print('=> Load after adding .net: ', k)
				replace_dict.append((k.replace('.net', ''), k))

		for k, k_new in replace_dict:
			sd[k_new] = sd.pop(k)
		keys1 = set(list(sd.keys()))
		keys2 = set(list(model_dict.keys()))
		set_diff = (keys1 - keys2) | (keys2 - keys1)
		print('#### Notice: keys that failed to load: {}'.format(set_diff))
		if args.dataset not in args.tune_from:  # new dataset
			print('=> New dataset, do not load fc weights')
			sd = {k: v for k, v in sd.items() if 'fc' not in k}
		if args.modality == 'Flow' and 'Flow' not in args.tune_from:
			sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
		model_dict.update(sd)
		model.load_state_dict(model_dict)

	cudnn.benchmark = True


	# Data loading code
	'''
	if args.modality != 'RGBDiff':
		normalize = GroupNormalize(input_mean, input_std)
	else:
		normalize = IdentityTransform()

	if args.modality == 'RGB':
		data_length = 1
	elif args.modality in ['Flow', 'RGBDiff']:
		data_length = 5
	'''

	# normalize
	normalize = GroupNormalize(input_mean, input_std)

	#load set
	train_loader = torch.utils.data.DataLoader(
		TSNDataSet(dataset=args.dataset, root_path=root_path, list_file=train_list, num_segments=args.num_segments,
							   new_length=1,
							   modality="RGB",
							   image_tmpl=prefix,
							   transform=torchvision.transforms.Compose([
								   train_augmentation,
								   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
								   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
								   normalize,
							   ]),
							   dense_sample=args.dense_sample),
	batch_size = args.batch_size, shuffle = True, drop_last = True,  # shuffle=True随机打乱
	num_workers = args.workers, pin_memory = True)

	val_loader = torch.utils.data.DataLoader(
		TSNDataSet(args.dataset, root_path, val_list,num_segments=args.num_segments,
				   new_length=1,
				   modality="RGB",
				   image_tmpl=prefix,
				   random_shift=False,
				   transform=torchvision.transforms.Compose([
					   GroupScale(int(scale_size)),
					   GroupCenterCrop(crop_size),
					   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
					   normalize,
				   ]),dense_sample=args.dense_sample),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)



	# define loss function (criterion) and optimizer
	# 交叉熵损失
	criterion = torch.nn.CrossEntropyLoss().cuda()

	optimizer = torch.optim.SGD(policies,
								args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)
	#evaluate
	if args.evaluate:
		log_training.write("***********Using Evaluate Mode***********\n")
		validate(val_loader, model, criterion, 0,log=log_training)
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
				'epoch': args.start_epoch,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'prec1': prec1,
				'best_prec1': best_prec1,
			}, is_best,epoch+1)
		return


	for epoch in range(args.start_epoch, args.epochs):
		log_training.write("[ " + str(datetime.now()).split(".")[0] + " ]\n")  # 打印时间
		log_training.write("********************************\n")
		log_training.write("EPOCH："+str(epoch+1)+"\n")
		# adjust learning rate
		adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)

		# evaluate on validation set
		if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
			prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training, tf_writer)

			# remember best prec@1 and save checkpoint
			is_best = prec1 > best_prec1
			best_prec1 = max(prec1, best_prec1)
			tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

			output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
			print(output_best)

			log_training.write(output_best)
			log_training.flush()

			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'prec1': prec1,
				'best_prec1': best_prec1,
			}, is_best,epoch+ 1)


		log_training.write("********************************\n")
	log_training.write("[ "+str(datetime.now()).split(".")[0]+" ]\n\n")  #print time

# train
def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()

	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		#input和target并不需要求梯度
		input = input.cuda(non_blocking = True)  #[4, 24, 224, 224]
		target = target.cuda(non_blocking = True)  #[4]

		#input_var = torch.autograd.Variable(input)
		#target_var = torch.autograd.Variable(target)

		# compute output
		output = model(input)
		loss = criterion(output, target)

		if args.debug and i==0:
			print("input.size()",input.size()) #[16,24,224,224]
			print("target.size()",target.size()) #[16]
			print("output.size()",output.size())

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output, target, topk=(1,5))
		#losses.update(loss.data[0], input.size(0))
		#top1.update(prec1[0], input.size(0))
		#top5.update(prec5[0], input.size(0))

		#torch1.7
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))
		top5.update(prec5.item(), input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		# 损失回传
		loss.backward()

		if args.clip_gradient is not None:  #梯度裁剪
			total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
			if total_norm > args.clip_gradient:
				print(("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm)))

		# parameter update
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0: #Print it every 20 epochs
			output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
						epoch+1, i+1, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
			print(output)
			log.write(output + '\n')
			log.flush()

	tf_writer.add_scalar('train/loss', losses.avg, epoch + 1)
	tf_writer.add_scalar('train/top1Accuracy', top1.avg, epoch + 1)
	tf_writer.add_scalar('train/top5Accuracy', top5.avg, epoch + 1)
	tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch + 1)

def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
			input = input.cuda(non_blocking = True)
			target = target.cuda(non_blocking = True)
			# compute output
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(output, target, topk=(1,5))

			# losses.update(loss.data[0], input.size(0))
			# top1.update(prec1[0], input.size(0))
			# top5.update(prec5[0], input.size(0))

			losses.update(loss.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))
			top5.update(prec5.item(), input.size(0))
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				output = ('Test: [{0}/{1}]\t'
					 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					  i+1, len(val_loader), batch_time=batch_time, loss=losses,
					 top1=top1, top5=top5))
				print(output)
				if log is not None:
					log.write(output + '\n')
					log.flush()

	output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
		  .format(top1=top1, top5=top5, loss=losses))
	print(output)

	#output_best = '\nBest Prec@1: %.3f'%(best_prec1)
	#print(output_best)

	if log is not None:
		log.write(output + '\n')
		#log.write(output + ' ' + output_best + '\n')
		log.flush()

	if tf_writer is not None:
		tf_writer.add_scalar('test/loss', losses.avg, epoch + 1)
		tf_writer.add_scalar('test/top1Accuracy', top1.avg, epoch + 1)
		tf_writer.add_scalar('test/top5Accuracy', top5.avg, epoch + 1)

	return top1.avg



def save_checkpoint(state, is_best,epoch):
	filename = os.path.join(checkpoint_dir,args.root_checkpoint, store_name+'_'+str(epoch)+'_checkpoint.pth.tar')
	if float(torch.__version__[:3])>=1.6:
		torch.save(state, filename, _use_new_zipfile_serialization=False)
	else:
		torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


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


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
	"""Sets the learning rate to the initial LR decayed by 10 """
	#print(lr_type)
	if lr_type == 'step_lr':
		decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
		lr = args.lr * decay
		decay = args.weight_decay

	elif lr_type == 'MAS_lr':
		if len(lr_steps)==4:
			decay = 0.1 ** (sum(epoch >= np.array(lr_steps[:2])))
			lr = args.lr * decay
			if epoch >= np.array(lr_steps[2]):
				lr = lr * 0.5
				if epoch >= np.array(lr_steps[3]):
					lr = lr * 0.2
		else:
			decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
			lr = args.lr * decay
		decay = args.weight_decay

	elif lr_type == 'cos_lr':
		import math
		lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
		decay = args.weight_decay
	else:
		raise NotImplementedError
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr * param_group['lr_mult']
		param_group['weight_decay'] = decay * param_group['decay_mult']

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
			correct_k = correct[:k].contiguous().view(-1).float().sum(0)  #torch1.7
			#correct_k = correct[:k].view(-1).float().sum(0,keepdim = True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

def check_rootfolders():
	"""Create log and model folder"""
	folders_util = [
					os.path.join("experiments", args.dataset, args.type, args.arch,
								 "num_segments_" + str(args.num_segments), args.store_name,
								 args.root_log),
					os.path.join("experiments", args.dataset, args.type, args.arch,
								 "num_segments_" + str(args.num_segments), args.store_name,
								 'tensorboardX'),
					os.path.join("experiments", args.dataset, args.type, args.arch,
								 "num_segments_" + str(args.num_segments), args.store_name,
								 args.root_checkpoint)
					]

	for folder in folders_util:
		if not os.path.exists(folder):
			print(('creating folder ' + folder))
			os.makedirs(folder)


if __name__ == '__main__':
	main()

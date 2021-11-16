from collections import OrderedDict
import os
import cv2
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision
from ops.models import TemporalModel
from ops.transforms import *
from opts import parser


def load_images(frame_dir, selected_frames, transform1, transform2):
    images = np.zeros((8, 224, 224, 3))
    orig_imgs = np.zeros_like(images)
    images_group = list()
    for i, frame_name in enumerate(selected_frames):
        im_name = os.path.join(frame_dir, frame_name)
        img = Image.open(im_name).convert('RGB')
        r_image = np.array(img)[:,:,::-1]
        images_group.append(img)
        orig_imgs[i],_ = transform2(([Image.fromarray(r_image)],1))

    torch_imgs,_ = transform1((images_group,1))
    return np.expand_dims(orig_imgs, 0), torch_imgs


def get_index(num_frames, num_segments):
    if num_frames > num_segments:
        tick = num_frames / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    else:
        offsets = np.zeros((num_segments,))
    return offsets + 1


def get_img(frame_dir, label):
    frame_names = os.listdir(frame_dir)
    frame_indices = get_index(len(frame_names), 8)
    selected_frames = ['{:05d}.jpg'.format(i) for i in frame_indices]

    RGB_vid, vid = load_images(frame_dir, selected_frames, transform1, transform2)

    return RGB_vid, vid


def get_heatmap(RGB_vid, vid, label):
    # get predictions, last convolution output and the weights of the prediction layer
    predictions = model(vid)
    layerout = model.base_model.layerout
    layerout = torch.tensor(layerout.numpy().transpose(0, 2, 3, 1))
    pred_weights = model.new_fc.weight.data.detach().cpu().numpy().transpose()

    pred = torch.argmax(predictions).item()

    cam = np.zeros(dtype=np.float32, shape=layerout.shape[0:3])
    for i, w in enumerate(pred_weights[:, label]):
        # Compute cam for every kernel
        cam += w * layerout[:, :, :, i].numpy()

    # Resize CAM to frame level
    cam = zoom(cam, (1, 32, 32))  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)

    # normalize
    cam -= np.min(cam)
    cam /= np.max(cam) - np.min(cam)

    heatmaps = []
    for i in range(0, cam.shape[0]):
        #   Create colourmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam[i]), cv2.COLORMAP_JET)

        # Create frame with heatmap
        heatframe = heatmap // 2 + RGB_vid[0][i] // 2
        heatmaps.append(heatframe[:, :, ::-1] / 255.)

    return heatmaps, pred

def show_cam(frame_dir, label):
    RGB_vid, vid = get_img(frame_dir=frame_dir , label=label)
    heatmaps, pred = get_heatmap(RGB_vid, vid, label=label)
    print("Visualizing for class\t{}-{}".format(label, category[label]))
    print(("MAS-Net predicted class\t{}-{}".format(pred, category[pred])))
    plt.rcParams['savefig.dpi'] = 250 #图片像素
    plt.rcParams['figure.dpi'] = 250 #分辨率
    plt.figure("origin")
    gs=gridspec.GridSpec(1,8)
    for i in range(1):
        for j in range(8):
            plt.subplot(gs[i,j])
            temp = RGB_vid[0][i*4+j]
            plt.imshow(temp[:,:,::-1]/255.)
            plt.axis('off')
    plt.title("Origin:Visualizing for class {}:{}".format(label, category[label]),fontsize=8,horizontalalignment="right")


    plt.rcParams['savefig.dpi'] = 250 #图片像素
    plt.rcParams['figure.dpi'] = 250 #分辨率
    plt.figure("predict")
    gs=gridspec.GridSpec(1,8)
    for i in range(1):
        for j in range(8):
            plt.subplot(gs[i,j])
            plt.imshow(heatmaps[i*8+j])
            plt.axis('off')
    plt.title('MAS-Net predicted class {}:{}'.format(pred, category[pred]),fontsize=8,horizontalalignment="right")
    plt.show()

def get_somethong_categories(imgpath):
    img_num = int(imgpath.split("/")[-1])
    ROOT_DATASET = "../"
    filename_imglist_train = ROOT_DATASET +'somethingv1/train.txt'
    filename_imglist_val = ROOT_DATASET +'somethingv1/valid.txt'
    with open(filename_imglist_train) as f1, open(filename_imglist_val) as f2:
        train_lines = f1.readlines()
        val_lines = f2.readlines()
    train_list = [item.rstrip() for item in train_lines]
    val_list = [item.rstrip() for item in val_lines]
    list = train_list+val_list
    for i in range(len(list)):
        if img_num==int(list[i].split(" ")[0]):
            label_num=int(list[i].split(" ")[-1])
            if i <len(train_list):
                dataset="train"
            else:
                dataset ="valid"
            print("It is in", dataset, "datasets")
            return label_num
        else:
            pass
    print("It is in test datasets")
    return 0



args = parser.parse_args()

label_file = '../somethingv1/category.txt'

category = []
for x in open(label_file):
    category.append(x.rstrip())
num_class = len(category)
print("category:",num_class)


weights = './experiments/test/v1_8/MASNet_somethingv1_resnet50_segment8_checkpoint.best.pth.tar'

model = TemporalModel(num_class, num_segments=args.num_segments, base_model=args.type, backbone=args.arch,
                      dropout=args.dropout)

checkpoint = torch.load(weights, map_location=torch.device('cpu'))
pretrained_dict = checkpoint['state_dict']
new_state_dict = OrderedDict()
for k, v in pretrained_dict.items():
    name = k[7:]  # remove 'module.'
    # name = name.replace('.net', '')
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()


# load image
crop_size = model.crop_size
scale_size = model.scale_size
input_mean = model.input_mean
input_std = model.input_std

transform1 = torchvision.transforms.Compose([
    GroupScale(int(scale_size)),
    GroupCenterCrop(crop_size),
    Stack(),
    ToTorchFormatTensor(),
    GroupNormalize(input_mean, input_std)
])
transform2 = torchvision.transforms.Compose([
    GroupScale(int(scale_size)),
    GroupCenterCrop(crop_size),
    Stack()
])

#5093 sm sm_sk
#5469 21598
num_video=21598

imgpath="../somethingv1/20bn-something-something-v1/"+str(num_video)
label_num= get_somethong_categories(imgpath)

show_cam(frame_dir=imgpath, label=label_num)
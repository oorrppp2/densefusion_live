import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet, vgg16_convs
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import cv2
import pyrealsense2 as rs


from yolact.data import COCODetection, get_label_map, MEANS, COLORS, cfg, set_cfg, set_dataset
from yolact.yolact import Yolact
from yolact.utils.augmentations import BaseTransform, FastBaseTransform, Resize
from yolact.layers.box_utils import jaccard, center_size, mask_iou
from yolact.utils import timer
from yolact.utils.functions import SavePath
from yolact.layers.output_utils import postprocess, undo_image_transformation
from collections import defaultdict


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--posecnn_model', type=str, default = '',  help='resume PoseCNN model')

# yolact args
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to evaulate model')
parser.add_argument('--fast_nms', default=True, type=str2bool,
                    help='Whether to use a faster, but not entirely correct version of NMS.')
parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                    help='Whether compute NMS cross-class or per-class.')
parser.add_argument('--display_masks', default=True, type=str2bool,
                    help='Whether or not to display masks over bounding boxes')
parser.add_argument('--display_bboxes', default=True, type=str2bool,
                    help='Whether or not to display bboxes around masks')
parser.add_argument('--display_text', default=True, type=str2bool,
                    help='Whether or not to display text (class [score])')
parser.add_argument('--display_scores', default=True, type=str2bool,
                    help='Whether or not to display scores in addition to classes')
parser.add_argument('--display', dest='display', action='store_true',
                    help='Display qualitative results instead of quantitative ones.')
parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                    help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                    help='In quantitative mode, the file to save detections before calculating mAP.')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='If display not set, this resumes mAP calculations from the ap_data_file.')
parser.add_argument('--max_images', default=-1, type=int,
                    help='The maximum number of images from the dataset to consider. Use -1 for all.')
parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                    help='The output file for coco bbox results if --coco_results is set.')
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                    help='The output file for coco mask results if --coco_results is set.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
parser.add_argument('--web_det_path', default='web/dets/', type=str,
                    help='If output_web_json is set, this is the path to dump detections into.')
parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                    help='Do not output the status bar. This is useful for when piping to a file.')
parser.add_argument('--display_lincomb', default=False, type=str2bool,
                    help='If the config uses lincomb masks, output a visualization of how those masks are created.')
parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                    help='Equivalent to running display mode but without displaying an image.')
parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                    help='Do not sort images by hashed image ID.')
parser.add_argument('--seed', default=None, type=int,
                    help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                    help='Outputs stuff for scripts/compute_mask.py.')
parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                    help='Do not crop output masks with the predicted bounding box.')
parser.add_argument('--image', default=None, type=str,
                    help='A path to an image to use for display.')
parser.add_argument('--images', default=None, type=str,
                    help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
parser.add_argument('--video', default=None, type=str,
                    help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
parser.add_argument('--video_multiframe', default=1, type=int,
                    help='The number of frames to evaluate in parallel to make videos play at higher fps.')
parser.add_argument('--score_threshold', default=0, type=float,
                    help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                    help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                    help='When displaying / saving video, draw the FPS on the frame')
parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                    help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                    shuffle=False,
                    benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                    display_fps=False,
                    emulate_playback=False)


opt = parser.parse_args()

color_cache = defaultdict(lambda: {})
# global opt

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
# cam_cx = 312.9869
# cam_cy = 241.3109
# cam_fx = 1066.778
# cam_fy = 1067.487
cam_cx = 318.892
cam_cy = 240.121
cam_fx = 614.678
cam_fy = 614.93
cam_scale = 0.001
K = [[cam_fx , 0 , cam_cx],
     [0, cam_fy, cam_cy],
     [0, 0, 1]]
# cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'
result_image = 'experiments/eval_result/ycb/image'

# color = [(255, 255, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
#               (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128),
#               (251, 194, 44), (240, 20, 134), (160, 103, 173), (70, 163, 210), (140, 227, 61),
#               (128, 128, 0), (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64)]

color = [[255, 255, 255], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0],
              [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128],
              [251, 194, 44], [240, 20, 134], [160, 103, 173], [70, 163, 210], [140, 227, 61],
              [128, 128, 0], [128, 0, 128], [0, 128, 128], [64, 0, 0], [0, 64, 0], [0, 0, 64]]
# for index, item in enumerate(color):
#     for i, t in enumerate(color[index]):
#         color[index][i] = float(t) / 255.0
class_name = ['master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can', 'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
              'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser', 'bowl', 'mug', 'power_drill', 'wood_block', 'scissors', 'large_marker',
              'large_clamp', 'extra_large_clamp', 'foam_brick']
def get_bbox(rmin, rmax, cmin, cmax):
    # rmin = int(posecnn_rois[idx][3]) + 1
    # rmax = int(posecnn_rois[idx][5]) - 1
    # cmin = int(posecnn_rois[idx][2]) + 1
    # cmax = int(posecnn_rois[idx][4]) - 1
    # print(str(rmin) + " , " + str(rmax) + " , " + str(cmin) + " , " + str(cmax))
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

# print("config")
# print(opt.config)

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

yolact = Yolact()
yolact.load_weights(opt.trained_model)
yolact.eval()
yolact.cuda()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
yolact.detect.use_fast_nms = opt.fast_nms
yolact.detect.use_cross_class_nms = opt.cross_class_nms

# evalimage(net, args.image)

import matplotlib.pyplot as plt


def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    # print("score_threshold : " + str(opt.score_threshold))

    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb=opt.display_lincomb,
                        crop_masks=opt.crop,
                        score_threshold=opt.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:opt.top_k]

        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].detach().cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(opt.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < opt.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if opt.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    if opt.display_fps:
        # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if opt.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if num_dets_to_consider == 0:
        return img_numpy

    if opt.display_text or opt.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if opt.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if opt.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if opt.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

    return masks.cuda().detach().cpu().numpy(), classes, boxes, img_numpy

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
corners = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    x = []
    y = []
    z = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        x.append(float(input_line[0]))
        y.append(float(input_line[1]))
        z.append(float(input_line[2]))
    input_file.close()
    cld[class_id] = np.array(cld[class_id])

    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    z_max = np.max(z)
    z_min = np.min(z)

    corners[class_id] = np.array([[x_min, y_min, z_min],
                                [x_max, y_min, z_min],
                                [x_max, y_max, z_min],
                                [x_min, y_max, z_min],

                                [x_min, y_min, z_max],
                                [x_max, y_min, z_max],
                                [x_max, y_max, z_max],
                                [x_min, y_max, z_max]])

    class_id += 1

# for now in range(500,2949):

save_path = "results/"
frame_i = 0

try:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)
    pipeline.start(config)

    while True:
        frame_i += 1
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # r, g, b = cv2.split(color_image)
        # color_image = cv2.merge((b, g, r))

        cv2.imshow('color_image', color_image)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # img = Image.open(folder_path + '/color_image' + str(now) + '.png')
        # depth = np.array(Image.open(folder_path + '/depth_image' + str(now) + '.png'))
        img = color_image
        depth = aligned_depth_image

        frame = torch.from_numpy(np.array(img)).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = yolact(batch)

        try:
            masks, classes, boxes, img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
        except:
            print("Yolact exception occur!")
            continue

        # print(classes)
        # print(boxes)
        #
        # plt.imshow(img_numpy)
        # plt.title("pred")
        # plt.show()


        """
            Image align
        """
        img = np.array(img)[:, :, :3]
        img.astype(np.float32)
        # image_mat0 = img[:,:,0].copy()
        # img[:,:,0] = img[:,:,2]
        # img[:,:,2] = image_mat0

        img_numpy = np.array(img_numpy)[:, :, :3]
        img_numpy.astype(np.float32)
        # img_numpy0 = img_numpy[:,:,0].copy()
        # img_numpy[:,:,0] = img_numpy[:,:,2]
        # img_numpy[:,:,2] = img_numpy0
        cv2.imshow("yolact detection", img_numpy)
        # cv2.imshow("yolact detection", img)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # else:
        #     continue

        # if not os.path.exists(save_path+str(scene_index)):
        #     os.mkdir(save_path+str(scene_index))

        # cv2.imwrite(save_path+str(scene_index)+"/"+str(now)+"_mask_result.png", img_numpy)

        my_result_wo_refine = []
        my_result = []
        masks_index = 0
        for index in range(len(classes)):
            itemid = classes[index] + 1
            box = boxes[index]
            try:
                rmin, rmax, cmin, cmax = get_bbox(box[1], box[3], box[0], box[2])
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = masks[index, :, :, 0]
                mask = mask_label * mask_depth

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                pt2 = depth_masked * cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

                pred_c = pred_c.view(bs, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                my_result_wo_refine.append(my_pred.tolist())

                for ite in range(0, iteration):
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                    my_mat = quaternion_matrix(my_r)
                    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                    my_mat[0:3, 3] = my_t

                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = refiner(new_cloud, emb, index)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)

                    my_mat_2[0:3, 3] = my_t_2

                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0:3, 3] = 0
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                    my_pred = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final

                model_points = cld[itemid]
                my_r = quaternion_matrix(my_r)[:3, :3]
                pred = np.dot(model_points, my_r.T) + my_t

                corner = corners[itemid]
                pred_box = np.dot(corner, my_r.T) + my_t
                transposed_pred_box = pred_box.T
                pred_box = transposed_pred_box/transposed_pred_box[2,:]
                pred_box_pixel = K @ pred_box
                pred_box_pixel = pred_box_pixel.astype(np.int64)

                cv2.line(img, (pred_box_pixel[0, 0], pred_box_pixel[1, 0]),
                       (pred_box_pixel[0, 1], pred_box_pixel[1, 1]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 1], pred_box_pixel[1, 1]),
                       (pred_box_pixel[0, 2], pred_box_pixel[1, 2]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 2], pred_box_pixel[1, 2]),
                       (pred_box_pixel[0, 3], pred_box_pixel[1, 3]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 3], pred_box_pixel[1, 3]),
                       (pred_box_pixel[0, 0], pred_box_pixel[1, 0]), (0,0,255), 2, lineType=cv2.LINE_AA)

                cv2.line(img, (pred_box_pixel[0, 4], pred_box_pixel[1, 4]),
                       (pred_box_pixel[0, 5], pred_box_pixel[1, 5]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 5], pred_box_pixel[1, 5]),
                       (pred_box_pixel[0, 6], pred_box_pixel[1, 6]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 6], pred_box_pixel[1, 6]),
                       (pred_box_pixel[0, 7], pred_box_pixel[1, 7]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 7], pred_box_pixel[1, 7]),
                       (pred_box_pixel[0, 4], pred_box_pixel[1, 4]), (0,0,255), 2, lineType=cv2.LINE_AA)

                cv2.line(img, (pred_box_pixel[0, 0], pred_box_pixel[1, 0]),
                       (pred_box_pixel[0, 4], pred_box_pixel[1, 4]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 1], pred_box_pixel[1, 1]),
                       (pred_box_pixel[0, 5], pred_box_pixel[1, 5]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 2], pred_box_pixel[1, 2]),
                       (pred_box_pixel[0, 6], pred_box_pixel[1, 6]), (0,0,255), 2, lineType=cv2.LINE_AA)
                cv2.line(img, (pred_box_pixel[0, 3], pred_box_pixel[1, 3]),
                       (pred_box_pixel[0, 7], pred_box_pixel[1, 7]), (0,0,255), 2, lineType=cv2.LINE_AA)

                transposed_pred = pred.T
                pred = transposed_pred/transposed_pred[2,:]
                pred_pixel = K @ pred
                pred_pixel = pred_pixel.astype(np.int64)

                _, cols = pred_pixel.shape
                del_list = []
                for i in range(cols):
                    if pred_pixel[0,i] >= img_length or pred_pixel[1,i] >= img_width :
                        del_list.append(i)
                pred_pixel = np.delete(pred_pixel, del_list, axis=1)

                img[pred_pixel[1,:], pred_pixel[0,:]] = color[int(itemid-1)]
                # img[pred_pixel[1,:], pred_pixel[0,:], 0] = color[itemid-1][0]
                # img[pred_pixel[1,:], pred_pixel[0,:], 1] = color[itemid-1][1]
                # img[pred_pixel[1,:], pred_pixel[0,:], 2] = color[itemid-1][2]

                # print("* " + class_name[itemid] + " *")
                # print("[pred box pixel]")
                # print(pred_box_pixel)
                # cv2.imshow("Predicted img ", img)
                # cv2.waitKey(0)

                # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

                my_result.append(my_pred.tolist())
            except (ZeroDivisionError, ValueError):
                print("exception...")
                # print("Yolact Detector Lost {0} at No.{1} keyframe".format(itemid, now))
                # my_result_wo_refine.append([0.0 for i in range(7)])
                # my_result.append([0.0 for i in range(7)])

        cv2.imshow("Predicted img ", img)
        # cv2.waitKey(0)
        # print("save img :: " + "results/scene"+str(scene_index)+"/"+str(now)+"_pose_result.png")

        # cv2.imshow('color_image', color_image)
        # cv2.imshow('aligned_depth_image', aligned_depth_image)
        key = cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        if key == ord('q'):
            break
        # elif cv2.waitKey(1) & 0xFF == ord('s'):
        elif key == ord('s'):
            print("Save image path : " + str(save_path + str(frame_i)+"_pose_result.png"))
            cv2.imwrite(save_path + str(frame_i)+"_mask_result.png", img_numpy)
            cv2.imwrite(save_path + str(frame_i)+"_pose_result.png", img)

    exit(0)
except Exception as e:
    print(e)
    pass


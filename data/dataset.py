import torch
import glob
import os
import cv2
import math
import time
from utils.augmentations import letterbox
import numpy as np
import warnings
from PIL import Image

class ParkingDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir='./parking_set/images', label_dir='./parking_set/five_freedom_labels', batch_size=8, backbone_img_size = 640):
        self.img_dir    = img_dir
        self.label_dir  = label_dir
        assert os.path.exists(img_dir),   "image directory does not exist"
        assert os.path.exists(label_dir), "label directory does not exist"
        self.imgs_jpg_path = glob.glob(img_dir+'/*')
        self.imgs_jpg_path.sort()
        self.lbs_txt_path = glob.glob(label_dir+'/*')
        self.lbs_txt_path.sort()
        assert os.path.exists(label_dir), "label directory does not exist"
        self.labels, list_shapes = [], []
        for img_path, lb_path in zip(self.imgs_jpg_path, self.lbs_txt_path):
            list_shapes.append(Image.open(img_path).size)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                numpy_lb = np.loadtxt(lb_path)
            if numpy_lb.size:
                self.labels.append(numpy_lb.reshape(-1, 6))
            else:
                self.labels.append(numpy_lb)
        self.shapes = np.array(list_shapes)
        self.max_num_targets_in_label = max([i.shape[0] for i in self.labels])

        self.backbone_img_size = backbone_img_size
        num_imgs = len(self.shapes)
        batch_index = np.floor(np.arange(num_imgs) / batch_size).astype(int)
        self.batch_index = batch_index
        self.num_imgs = num_imgs
        self.img_index = range(num_imgs)
    
    def __len__(self):
        assert len(self.imgs_jpg_path) == len(self.lbs_txt_path)
        return len(self.imgs_jpg_path)

    def __getitem__(self, index):
        index = self.img_index[index]
        img_path, img, size_ratio = self.load_image(index)
        labels = self.labels[index].copy()
        if labels.size:
            labels[:, 1:] = labels[:, 1:] * size_ratio
        num_targets_per_label = len(labels)
        # 10 == index + cls + four points coordinate
        # labels_out = torch.zeros((num_targets_per_label, 10))
        # if num_targets_per_label:
        #     labels_out[:, 1:] = torch.from_numpy(labels)
        labels_out = torch.zeros((self.max_num_targets_in_label, 6))
        if num_targets_per_label:
            labels_out[:num_targets_per_label, :] = torch.from_numpy(labels)
        numpy_img = np.array(img).transpose((2, 0, 1))
        numpy_img = np.ascontiguousarray(numpy_img)
        return img_path, torch.from_numpy(numpy_img), labels_out

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        img_path = self.imgs_jpg_path[i]
        img = Image.open(img_path) #RGB
        h0, w0 = img.size #self.shapes[i]
        assert h0 == w0,   "image is not square"
        size_ratio = self.backbone_img_size / h0
        if size_ratio != 1:
            img = img.resize((self.backbone_img_size, self.backbone_img_size), Image.LANCZOS)
        return img_path, img, size_ratio
    
    @staticmethod
    def collate_fn(batch):
        imgs_path, ims, labels = zip(*batch)  # transposed
        # for i, lb in enumerate(labels):
        #     lb[:, 0] = i
        stacked_ims, concated_label = torch.stack(ims, 0), torch.stack(labels, 0)
        return imgs_path, stacked_ims, concated_label
    
class LoadVideo:
    def __init__(self, video_path='car.mp4', img_size=(640,640), vid_stride=1):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.img_size = img_size
        self.vid_stride = vid_stride  # video frame-rate stride
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        self.cap = cap
        # Start thread to read frames from video stream
        assert cap.isOpened(), f'Failed to open {video_path}'
        fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
        self.n = 0
        self.num_frame = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        _, self.img = cap.read()  # guarantee first frame

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        ##############################################################
        if self.cap.isOpened() and self.n < self.num_frame:
            self.n += 1
            self.cap.grab()  # .read() = .grab() followed by .retrieve()
            if self.n % self.vid_stride == 0:
                success, im = self.cap.retrieve()
                if success:
                    self.img = im
                else:
                    self.img = np.zeros_like(self.img)
                    self.cap.open(self.video_path)  # re-open stream if signal was lost
            time.sleep(0.2)  # wait time
        ##############################################################
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        im0 = self.img.copy()
        im0 = letterbox(im0, self.img_size)
        
        im = letterbox(im0, self.img_size)  # resize
        im = im[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
        im = np.ascontiguousarray(im)  # contiguous

        return self.video_path, im, im0

    def __len__(self):
        return 1  # 1E12 frames = 32 streams at 30 FPS for 30 years
    
if __name__ == '__main__':
    dataset = LoadVideo(video_path='./parking_set/car.mp4')
    for path, img, img0 in dataset:
        print("img0 ",img0.shape)
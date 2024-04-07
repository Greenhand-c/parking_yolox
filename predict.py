import torch
import cv2
import numpy as np
from models.yolox import YOLOX
from models.pa_fpn import YOLOPAFPN
from models.det_head import YOLOXHead
from data.dataset import LoadVideo
from data.dataset import ParkingDataset
from data.dataloader import InfiniteDataLoader
from utils.general import non_max_suppression
from utils.visualize import visualize_assign
from utils.points import coordinate_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def video_predict_fn(model, dataset):
    for path, img, img0 in dataset:
        video_img = torch.Tensor(img).to(device) #[3,640,640]
        video_img = video_img.unsqueeze(dim=0) #[1,3,640,640]
        video_img = video_img.float()
        predictions = model(video_img) #[1,8400,10]
        predictions = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)
        visualize_assign(img0, predictions, input_path=path, output_path='./parking_set/output_video.mp4')

def image_predict_fn(model, pred_loader):
    for imgs_path, imgs_outs, labels_outs in pred_loader:
        imgs0 = imgs_outs.detach()
        imgs_outs = imgs_outs.float()
        '''
        for bs, det in enumerate(labels_outs):
            img = imgs0[bs].numpy().transpose((1, 2, 0))
            img = np.ascontiguousarray(img)
            img_name = imgs_path[bs].split('/')[-1]
            out_path = './predict_img/' + img_name
            visualize_polygons = []
            parking_boxes = det[..., 1:6].detach()
            if all(parking_boxes[0] == 0):
                visualize_polygons = None
            else:
                for box in parking_boxes:
                    if any(box):
                        point4, point1, length = box[:2], box[2:4], box[4]
                        point2 = coordinate_transform(point4, point1, length)
                        point3 = np.array([(point2[0]+point4[0]-point1[0]),(point2[1]+point4[1]-point1[1])])
                        p = np.concatenate((point1.reshape(1, 2), point2.reshape(1, 2), point3.reshape(1, 2), point4.reshape(1, 2)), axis=0)
                        visualize_polygons.append(p)
            if visualize_polygons is not None:
                for polygon_points in visualize_polygons:
                    hull = cv2.convexHull(polygon_points.astype(np.float32))
                    cv2.polylines(img, [hull.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imwrite(out_path, img)
            print(f"--> waiting for save the {img_name} image")
        '''
        outputs = model(imgs_outs.to(device))
        outputs = non_max_suppression(outputs, conf_thres=0.2, iou_thres=0.45)
        for bs, det in enumerate(outputs):
            img = imgs0[bs].numpy().transpose((1, 2, 0))
            img = np.ascontiguousarray(img)
            img_name = imgs_path[bs].split('/')[-1]
            out_path = './predict_img/' + img_name
            visualize_polygons = []
            parking_boxes = det[..., :5].detach()
            if parking_boxes.shape[0] == 0:
                visualize_polygons = None
            else:
                for box in parking_boxes:
                    if any(box):
                        point4, point1, length = box[:2], box[2:4], box[4]
                        point2 = coordinate_transform(point4, point1, length).cpu().numpy()
                        point4, point1 = point4.cpu().numpy(), point1.cpu().numpy()
                        point3 = np.array([(point2[0]+point4[0]-point1[0]),(point2[1]+point4[1]-point1[1])])
                        p = np.concatenate((point1.reshape(1, 2), point2.reshape(1, 2), point3.reshape(1, 2), point4.reshape(1, 2)), axis=0)
                        visualize_polygons.append(p)
            if visualize_polygons is not None:
                for polygon_points in visualize_polygons:
                    hull = cv2.convexHull(polygon_points)
                    cv2.polylines(img, [hull.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imwrite(out_path, img)
            print(f"--> waiting for save the {img_name} image")

def main():
    bkb = YOLOPAFPN(depth=0.33, width=0.5).to(device)
    head = YOLOXHead(num_classes=1, width=0.5).to(device)
    yolox = YOLOX(backbone=bkb, head=head).eval().to(device)
    '''
    dataset = LoadVideo(video_path='./parking_set/car.mp4', img_size=(640,640), vid_stride=1)
    print("--> loading weight")
    checkpoint = torch.load('./weights/weight.pth.tar')
    yolox.load_state_dict(checkpoint["state_dict"])
    video_predict_fn(yolox, dataset)
    '''
    pred_dataset = ParkingDataset(img_dir='./parking_set/images',
                                   label_dir='./parking_set/five_freedom_labels',
                                   batch_size=4,
                                   backbone_img_size = 640)
    pred_dataloader = InfiniteDataLoader(dataset=pred_dataset,
                                          batch_size=4,
                                          pin_memory=True,
                                          collate_fn=ParkingDataset.collate_fn,
                                          shuffle=False,
                                          drop_last=False)
    print("--> loading weight")
    checkpoint = torch.load('./weights/five_weight.pth.tar')
    yolox.load_state_dict(checkpoint["state_dict"])
    image_predict_fn(yolox, pred_dataloader)

if __name__ == "__main__":
    main()
import torch.nn as nn
import torch
import shapely
import numpy as np
from shapely.geometry import Polygon, MultiPoint

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def bbox_iou_eval(self, box1, box2):
        combined_coords = torch.cat((box1, box2), dim=0)
        box1 = np.array(box1.detach().cpu()).reshape(-1, 2)
        poly1 = Polygon(box1).convex_hull #POLYGON
        box2 = np.array(box2.detach().cpu()).reshape(-1, 2)
        poly2 = Polygon(box2).convex_hull
        if not poly1.intersects(poly2):  # If two quadrilaterals do not intersect
            iou = 0
        else:
            try:
                inter_area = poly1.intersection(poly2).area  # intersect area
                iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured, iou set to 0')
                iou = 0
        if self.loss_type == "giou":
            multi_point = MultiPoint(combined_coords)
            min_bounding_rect = multi_point.minimum_rotated_rectangle
            min_bounding_rect_area = min_bounding_rect.area
            if not poly1.intersects(poly2):
                giou = -1 + (poly1.area + poly2.area) / min_bounding_rect_area
            else:
                try:
                    giou = iou - (min_bounding_rect_area - (poly1.area + poly2.area - inter_area)) / min_bounding_rect_area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured, giou set to -1')
                    giou = -1
            return giou
        return iou
    
    def equal_bboxes_iou(self, pred, target):
        # calculate loss: return shape ==> [tensor1.shape[0]]
        assert pred.shape[0] == target.shape[0]
        iou_per_image = torch.zeros(pred.shape[0])
        for i in range(pred.shape[0]):
            pred_coor = pred[i].reshape(-1, 2)
            target_coor = target[i].reshape(-1, 2)
            iou_per_image[i] = self.bbox_iou_eval(pred_coor, target_coor)
        return iou_per_image
    
    def forward(self, pred, target):
        if self.loss_type == "iou":
            iou = self.equal_bboxes_iou(pred, target)
            loss = 1 - iou
        elif self.loss_type == "giou":
            giou = self.equal_bboxes_iou(pred, target)
            # -1 < giou < 1 == 0 < loss < 2
            loss = 1 - giou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
    
    def calculate_triangle_area(self, t1, t2, t3):
        # Given three vertices, find the area of the triangle
        area = 0.5 * abs(t1[0] * (t2[1] - t3[1]) + t2[0] * (t3[1] - t1[1]) 
                        + t3[0] * (t1[1] - t2[1]))
        return area

    def calculate_area(self, poly):
        # calculate quadrilateral area
        area = self.calculate_triangle_area(poly[0],poly[1],poly[2])
        area += self.calculate_triangle_area(poly[0],poly[1],poly[3])
        area += self.calculate_triangle_area(poly[0],poly[2],poly[3])
        area += self.calculate_triangle_area(poly[1],poly[2],poly[3])
        return abs(area) / 2

if __name__ == '__main__':
    # box = [四个点的坐标，顺序无所谓]
    #box1 = [np.array([0, 0]), np.array([1, 1]), np.array([0, 1]), np.array([1, 0])]
    box1 = torch.tensor([[0, 0], [1, 1], [0, 1], [1, 0]])
    #box2 = [np.array([1, 1]), np.array([2, 0]), np.array([3, 1]), np.array([2.5, 2])]
    box2 = torch.tensor([[1, 1], [2, 0], [3, 1], [2.5, 2]])
    iouloss = IOUloss(loss_type="giou")
    giou = iouloss.bbox_iou_eval(box1, box2)
    print(giou)

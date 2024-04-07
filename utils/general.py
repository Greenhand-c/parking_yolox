import torch
from utils.bboxes import bboxes_iou
from utils.points import four_points

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Args:
        prediction: tensor[bs, 8400, 5 + conf + cls]
    Returns:
        list of batch size, on (n, 7) tensor per image [5, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 5] > conf_thres  # candidates
    max_nms = 30000  # maximum number of boxes

    # Settings
    output = [torch.zeros((0, 7), device=prediction.device)] * bs
    for i, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[i]]  # confidence filter
        x = x[x[:, 5].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
        while x.numel() > 0:
            output[i] = torch.cat((output[i], x[0:1]))
            iou = bboxes_iou(four_points(x[0:1, 0:5]), four_points(x[:, 0:5]))
            xi = iou <= iou_thres
            x = x[xi[0]]

    return output
import torch
import importlib

def bboxes_iou(target, pred):
    # calculate iou: return shape ==> [target.shape[0], pred.shape[0]]
    iou_module = importlib.import_module("models.iou_samples")
    iou_per_box = iou_module.IOUloss(loss_type="iou")
    output = torch.zeros(target.shape[0], pred.shape[0])
    for i in range(target.shape[0]):
        target_coor = target[i].reshape(-1, 2)
        for j in range(pred.shape[0]):
            pred_coor = pred[j].reshape(-1, 2)
            output[i][j] = iou_per_box.bbox_iou_eval(target_coor, pred_coor)
    return output
            
if __name__ == '__main__':
    box1 = torch.tensor([[0, 0, 2, 2, 2, 0, 0, 2], [1, 1, 1, 3, 3, 3, 3, 1]])
    box2 = torch.tensor([[0, 0, 2, 2, 2, 0, 0, 2], [1, 1, 1, 3, 3, 3, 3, 1]])
    loss = bboxes_iou(box1,box2)
    print(loss)
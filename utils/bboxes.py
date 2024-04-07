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

def cos_distance(target, pred):
    '''
    Args:
        target: gt boxes [num_targets, 5]
        pred: pred boxes [num_anchors, 5]
    Return:
    if num_targets == num_anchors:
        cos_distance: cosine similarity [num_targets]
    if num_targets != num_anchors:
        cos_distance: cosine similarity [num_targets, num_anchors]
    '''
    if target.shape[0] == pred.shape[0]:
        cos_distance = torch.cosine_similarity(target, pred, eps=1e-8)
        return cos_distance
    else:
        cos_distance = torch.cosine_similarity(target.unsqueeze(1),pred.unsqueeze(0),dim=-1,eps=1e-8)
        return cos_distance

def standard_euclidean_distance(target, pred):
    '''
    Args:
        target: gt point3 coordinate [num_targets, 2]
        pred: pred point3 coordinate [num_anchors, 2]
    '''
    if target.shape[0] != pred.shape[0]:
        eudt = torch.zeros(target.shape[0], pred.shape[0])
        for i in range(target.shape[0]):
            cov_x = torch.cov(torch.cat((target[i:i+1,0],pred[:,0])))
            cov_y = torch.cov(torch.cat((target[i:i+1,1],pred[:,1])))
            eudt[i] = torch.sqrt(torch.pow((target[i,0]-pred[:,0]),2)/cov_x + \
                                 torch.pow((target[i,1]-pred[:,1]),2)/cov_y)
        return eudt
    else:
        eudt = torch.zeros(target.shape[0])
        for i in range(target.shape[0]):
            cov_x = torch.cov(torch.cat((target[:,0],pred[:,0])))
            cov_y = torch.cov(torch.cat((target[:,1],pred[:,1])))
            eudt[i] = torch.sqrt(torch.pow((target[i,0]-pred[i,0]),2)/cov_x + \
                                 torch.pow((target[i,1]-pred[i,1]),2)/cov_y)
        return eudt

            
if __name__ == '__main__':
    box1 = torch.tensor([[0, 0, 2, 2, 2, 0, 0, 2]])
    box2 = torch.tensor([[0, 0, 2, 2, 2, 0, 0, 2], [1, 1, 1, 3, 3, 3, 3, 1]])
    #iou = bboxes_iou(box1,box2)
    #print((iou<0.2).shape)
    target = torch.arange(20).view(-1,5).float()+1.
    pred = torch.arange(15).view(-1,5).float()+11.
    #eu = standard_euclidean_distance(target, pred)
    cos = cos_distance(target,pred)
    print(target)
    print(pred)
    print(cos)
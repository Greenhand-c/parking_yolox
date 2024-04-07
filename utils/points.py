import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def find_center(gt_bboxes_per_image):
    """
    This function is to find the coordinates of 
    the center point of the polygon in the dataset.
    input: 
        gt_bboxes_per_image ==> tensor[num_targets, 2*2+1]
    Return: 
        center_coordinates  ==> tensor[num_targets, 2]
        targets_point3      ==> tensor[num_targets, 2]
    """
    center_coordinates = torch.zeros(gt_bboxes_per_image.shape[0], 2)
    targets_point3 = torch.zeros((gt_bboxes_per_image.shape[0], 2))
    for i in range(gt_bboxes_per_image.shape[0]):
        point1 = gt_bboxes_per_image[i,0:2]
        point2 = gt_bboxes_per_image[i,2:4]
        length = gt_bboxes_per_image[i,4]
        point3 = coordinate_transform(point1, point2, length)
        targets_point3[i] = point3
        center_coordinates[i, 0] = (gt_bboxes_per_image[i,0:2][0]+point3[0])/2
        center_coordinates[i, 1] = (gt_bboxes_per_image[i,0:2][1]+point3[1])/2
        '''
        center_coordinates[i, 0] = (gt_bboxes_per_image[i,0:2][0]+
                                    gt_bboxes_per_image[i,2:4][0])/2
        center_coordinates[i, 1] = (gt_bboxes_per_image[i,0:2][1]+
                                    gt_bboxes_per_image[i,2:4][1])/2
        '''
    return center_coordinates, targets_point3

def four_points(bboxes_per_image):
    """
    This function is to find the coordinates of 
    the center point of the polygon in the dataset.
    input: 
        bboxes_per_image ==> tensor[num_targets, 2*2+1]
    Return: 
        four_points      ==> tensor[num_targets, 2*4]
    """
    four_points = torch.zeros((bboxes_per_image.shape[0], 8))
    for i in range(bboxes_per_image.shape[0]):
        point1 = bboxes_per_image[i,0:2]
        point2 = bboxes_per_image[i,2:4]
        length = bboxes_per_image[i,4]
        point3 = coordinate_transform(point1, point2, length)
        point4 = torch.tensor([(point1[0]+point3[0]-point2[0]),(point1[1]+point3[1]-point2[1])])
        point4 = point4.to(device)
        four_points[i] = torch.cat((point1,point2,point3,point4))

    return four_points

def coordinate_transform(point1, point2, length):
    '''
        (x3 - x2) = -s*(y2 - y1)
        (y3 - y2) =  s*(x2 - x1)
    '''
    vector12 = point2 - point1
    square_l12 = (point2[0]-point1[0]).pow(2) + (point2[1]-point1[1]).pow(2)
    if square_l12 < 1e-8:
        s = torch.sqrt(length.pow(2) / 1e-8)
    else:
        s = torch.sqrt(length.pow(2) / square_l12)
    vector23 = torch.tensor([-s*(point2[1]-point1[1]), s*(point2[0]-point1[0])])
    vector23 = vector23.to(device)
    z_zero = torch.tensor([0.]).to(device)
    z_vector12 = torch.cat((vector12, z_zero))
    z_vector23 = torch.cat((vector23, z_zero))
    cross_product1 = torch.cross(z_vector12, z_vector23)[-1]
    if cross_product1 < 0:
        return vector23 + point2
    else:
        return -vector23 + point2

if __name__ == "__main__":
    point4 = torch.tensor([636.2770343099704, 40.86818783114816])
    point1 = torch.tensor([633.3351200227546, 445.768901819366])
    length = torch.tensor([160.5892238770438])
    gt_bboxes_per_image = torch.tensor([[636.2770343099704, 40.86818783114816, 633.3351200227546, 445.768901819366, 160.5892238770438]])
    point2 = coordinate_transform(point4, point1, length)
    c = np.array([(point2[0]+point4[0])/2,(point2[1]+point4[1])/2])
    center = find_center(gt_bboxes_per_image)
    print(point2)
    print(c)
    print(center)
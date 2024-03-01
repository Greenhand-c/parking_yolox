import torch

def find_center(gt_bboxes_per_image):
    """
    This function is to find the coordinates of 
    the center point of the polygon in the dataset.
    input: gt_bboxes_per_image shape ==> tensor[num_targets, num_points*2]
    Return: center_coordinates shape ==> tensor[num_targets, 2]
    """
    center_coordinates = torch.zeros(gt_bboxes_per_image.shape[0], 2)
    for i in range(gt_bboxes_per_image.shape[0]):
        center_coordinates[i, 0] = (torch.max(gt_bboxes_per_image[i, 0::2])+
                                    torch.min(gt_bboxes_per_image[i, 0::2]))/2
        center_coordinates[i, 1] = (torch.max(gt_bboxes_per_image[i, 1::2])+
                                    torch.min(gt_bboxes_per_image[i, 1::2]))/2
    return center_coordinates

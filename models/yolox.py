import torch
import torch.nn as nn

from models.det_head import YOLOXHead
from models.pa_fpn import YOLOPAFPN

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(1)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, corr_loss, conf_loss, cls_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "corr_loss": corr_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)

if __name__ == '__main__':
    images = torch.zeros(1, 3, 640, 640)
    # labels shape ==> [bs, num_targets, cls+four_points]
    labels = torch.tensor([[[0,62.3,86.4,76.7,86.9,78.2,44.5,63.3,44.5],
                            [0,63.3,44.5,78.2,44.5,79.6,4.1,63.6,4.0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]],
                           [[0,63.7,64.3,77.1,63.2,76.9,22.9,62.5,23.0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0]]])
    bkb = YOLOPAFPN(depth=0.33, width=0.5)
    head = YOLOXHead(1,4,0.5)
    yolox = YOLOX(backbone=bkb,head=head)#.eval() #train()
    outputs = yolox(images,labels)
    print(yolox)#["total_loss"])
    print(outputs)
import os
import time
import torch

from tqdm import tqdm
import torch.optim as optim
from models.yolox import YOLOX
from models.pa_fpn import YOLOPAFPN
from models.det_head import YOLOXHead
from data.dataset import ParkingDataset
from data.dataloader import InfiniteDataLoader

load_weight = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

def save_checkpoint(state, checkpoint='./weights/five_freedom_weight.pth.tar'):
    print("--> saving checkpoint")
    torch.save(state, checkpoint)

def load_checkpoint(checkpoint, model, optimizer):
    print("--> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def train_fn(num_epochs, train_loader, model, optimizer):
    avg_loss_pre_epoch = 10000.0
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, leave=True)
        mean_loss = []
        for batch_idx, (_, imgs_outs, labels_outs) in enumerate(loop):
            imgs_outs, labels_outs = imgs_outs.float(), labels_outs.float()
            output = model(imgs_outs.to(device), labels_outs.to(device))
            loss = output["total_loss"]
            corr_loss = output["corr_loss"]
            conf_loss = output["conf_loss"]
            cls_loss = output["cls_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean_loss.append(loss.item())
            loop.set_postfix(loss = f"[corr_loss:{corr_loss},conf_loss:{conf_loss},cls_loss:{cls_loss}]")
            loop.set_description()
        avg_loss = sum(mean_loss)/len(mean_loss)
        print(f"{epoch}th training epoch average loss: {avg_loss}")
        if avg_loss < avg_loss_pre_epoch:
            checkpoint = {"state_dict": model.state_dict(),"optimizer": optimizer.state_dict()}
            save_checkpoint(checkpoint)
            avg_loss_pre_epoch = avg_loss
            time.sleep(3)

def main():
    bkb = YOLOPAFPN(depth=0.33, width=0.5)
    head = YOLOXHead(num_classes=1, width=0.5)
    yolox = YOLOX(backbone=bkb, head=head).to(device)
    optimizer = optim.Adam(yolox.parameters(), lr = 2e-5, weight_decay=0)
    train_dataset = ParkingDataset(img_dir='./parking_set/images',
                                   label_dir='./parking_set/five_freedom_labels',
                                   batch_size=4,
                                   backbone_img_size = 640)
    train_dataloader = InfiniteDataLoader(dataset=train_dataset,
                                          batch_size=4,
                                          pin_memory=True,
                                          collate_fn=ParkingDataset.collate_fn,
                                          shuffle=False,
                                          drop_last=False)
    if os.path.exists('./weights/five_freedom_weight.pth.tar') and load_weight:
        load_checkpoint(checkpoint=torch.load('./weights/five_freedom_weight.pth.tar'), model=yolox, optimizer=optimizer)
    train_fn(num_epochs=2000, train_loader=train_dataloader, model=yolox, optimizer=optimizer)

if __name__ == "__main__":
    main()
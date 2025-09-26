
import torch, timm, pytorch_lightning as pl
import torch.nn as nn
import torchmetrics

NUM_CLASSES=5

class ImgClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4, wd=1e-4, class_weights=None, model_name="tf_efficientnet_b0"):
        super().__init__()
        self.save_hyperparameters()
        self.net = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
        self.crit = nn.CrossEntropyLoss(weight=class_weights)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")

    def forward(self,x): return self.net(x)

    def step(self, batch, stage):
        y = batch["label"]; yhat = self.forward(batch["image"])
        loss = self.crit(yhat, y)
        preds = yhat.argmax(1)
        f1 = self.f1(preds, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        return loss

    def training_step(self,batch,_): return self.step(batch,"train")
    def validation_step(self,batch,_): return self.step(batch,"val")
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return {"optimizer":opt,"lr_scheduler":sch}

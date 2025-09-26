
import torch, torch.nn as nn, pytorch_lightning as pl
import torchmetrics

class TabClassifier(pl.LightningModule):
    def __init__(self, in_dim, lr=1e-3, wd=1e-4, hidden=128):
        super().__init__()
        self.save_hyperparameters()
        self.m = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, 5)
        )
        self.crit = nn.CrossEntropyLoss()
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=5, average="macro")

    def forward(self,x): return self.m(x)
    def step(self,b,stage):
        y=b["y"]; yhat=self.forward(b["x"]); loss=self.crit(yhat,y)
        preds=yhat.argmax(1); f1=self.f1(preds,y)
        self.log(f"{stage}_loss",loss,prog_bar=True); self.log(f"{stage}_f1",f1,prog_bar=True)
        return loss
    def training_step(self,b,_): return self.step(b,"train")
    def validation_step(self,b,_): return self.step(b,"val")
    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=self.hparams.lr,weight_decay=self.hparams.wd)
        return opt

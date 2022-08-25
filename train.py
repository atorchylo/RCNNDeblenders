import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
import pytorch_lightning as pl
# model imports
from config import BackboneConfig, RPNConfig, ROIBoxHeadConfig
from models.FasterRCNN import FasterRCNN
# data imports
from data import DatasetBoxes, collate_fn, CollatedBatchType
from torch.utils.data import DataLoader
from utils.plot_utils import plot_batch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Dict, Optional, List, Tuple


class Model(pl.LightningModule):
    def __init__(
            self,
            rcnn_model,
            train_dataset,
            valid_dataset,
            num_workers=0,
            batch_size=32,
            learning_rate=5e-5
    ):
        super().__init__()
        # set up model and datasets
        self.rcnn_model = rcnn_model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # set up optimization parameters
        self.metric = MeanAveragePrecision()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
                ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        return self.rcnn_model(images, targets)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.rcnn_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.0005)

    def training_step(self, batch: CollatedBatchType, batch_idx: int = 0) -> torch.Tensor:
        images, targets = batch
        loss_dict = self(images, targets)
        self.log('loss/box_reg', loss_dict['loss_box_reg'], prog_bar=True, logger=True)
        self.log('loss/classifier', loss_dict['loss_classifier'], prog_bar=True, logger=True)
        self.log('loss/rpn_objectness', loss_dict['loss_objectness'], prog_bar=True, logger=True)
        self.log('loss/rpn_box_reg', loss_dict['loss_rpn_box_reg'], prog_bar=True, logger=True)
        loss = sum(loss_dict.values())
        self.log("loss/total", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: CollatedBatchType, batch_idx: int = 0):
        images, targets = batch
        preds = self(images)
        self.metric.update(preds, targets)
        if batch_idx == 0:
            plot_img = plot_batch(images, targets, preds, num=5)
            self.logger.experiment.add_image('images', plot_img, self.current_epoch, dataformats="HWC")

    def validation_epoch_end(self, outputs):
        map_metrics = self.metric.compute()
        for name, value in map_metrics.items():
            self.log(f'map/{name}', value, logger=True, on_epoch=True)
        self.metric.reset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                          collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                          collate_fn=collate_fn)


def parse():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        Generates cosmos HSC data for training. 
        """), formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to the train dataset.")

    parser.add_argument("--valid_path", type=str, required=True,
                        help="Path to te validation dataset.")

    parser.add_argument("--save_path", type=str, default='./',
                        help="Path to logging directory")

    parser.add_argument("--train_gpu", type=bool, default=False,
                        help="Train on max number of GPU")

    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")

    parser.add_argument("--train_mps", type=bool, default=False,
                        help="Train on max number of Apple’s Metal Performance Shaders (MPS)")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training")

    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="learning rate for training")

    parser.add_argument("--optimize_bs", type=bool, default=False,
                        help="Finds the optimal batch size for training")

    parser.add_argument("--optimize_lr", type=bool, default=False,
                        help="Finds the optimal batch size for training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    # load rcnn
    rcnn_model = FasterRCNN(backbone_config=BackboneConfig()._to_dict(),
                            rpn_config=RPNConfig()._to_dict(),
                            roi_box_head_config=ROIBoxHeadConfig()._to_dict())

    # load data
    train_dataset = DatasetBoxes(args.train_path)
    valid_dataset = DatasetBoxes(args.valid_path)

    # wrap in pt_lightning model
    model = Model(rcnn_model, train_dataset, valid_dataset,
                  num_workers=args.num_workers,
                  batch_size=args.batch_size,
                  learning_rate=args.learning_rate)

    # train!
    auto_scale_batch_size = True if args.optimize_bs else None
    auto_lr_find = True if args.optimize_lr else None
    if args.train_gpu:
        accelerator = 'gpu'
        devices = -1
    elif args.train_mps:
        accelerator = 'mps'
        devices = -1
    else:
        accelerator = 'cpu'
        devices = 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=5,
        val_check_interval=1.0,
        max_epochs=100,
        default_root_dir=args.save_path,
        auto_scale_batch_size=auto_scale_batch_size,
        auto_lr_find=auto_lr_find,
    )

    if args.optimize_bs or args.optimize_lr:
        trainer.tune(model)
    trainer.fit(model)

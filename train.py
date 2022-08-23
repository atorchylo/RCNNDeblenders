from src.models.FasterRCNN import FasterRCNN
from src.data import DatasetBoxes, collate_fn
from torch.utils.data import DataLoader

from config import BackboneConfig, RPNConfig, ROIBoxHeadConfig
import pytorch_lightning as pl

TRAIN_DATASET = 'raw_data/cosmos_HSC/train'
VALID_DATASET = 'raw_data/cosmos_HSC/valid'

# load model
model = FasterRCNN(backbone_config=BackboneConfig()._to_dict(),
                   rpn_config=RPNConfig()._to_dict(),
                   roi_box_head_config=ROIBoxHeadConfig()._to_dict())

# load data
train_dataset = DatasetBoxes(TRAIN_DATASET)
valid_dataset = DatasetBoxes(VALID_DATASET)
train_dl = DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)
valid_dl = DataLoader(valid_dataset, batch_size=10, collate_fn=collate_fn)

# train!
trainer = pl.Trainer(log_every_n_steps=1)  # for validation: val_check_interval=1.0, )
trainer.fit(model, train_dl, valid_dl)

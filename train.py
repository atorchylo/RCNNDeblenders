from models.FasterRCNN import FasterRCNN
from data import DatasetBoxes, collate_fn
from torch.utils.data import DataLoader

from config import BackboneConfig, RPNConfig, ROIBoxHeadConfig
import pytorch_lightning as pl

if __name__ == "__main__":
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
    args = parser.parse_args()

    # load model
    model = FasterRCNN(backbone_config=BackboneConfig()._to_dict(),
                       rpn_config=RPNConfig()._to_dict(),
                       roi_box_head_config=ROIBoxHeadConfig()._to_dict())

    # load data
    train_dataset = DatasetBoxes(args.train_path)
    valid_dataset = DatasetBoxes(args.valid_path)
    train_dl = DataLoader(train_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=10, collate_fn=collate_fn)

    # train!
    trainer = pl.Trainer(log_every_n_steps=5, max_epochs=100, default_root_dir=args.save_path)  # for validation: val_check_interval=1.0, )
    trainer.fit(model, train_dl, valid_dl)

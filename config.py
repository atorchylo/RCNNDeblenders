"""Config setup architecture"""
# TODO refactor
import json



class BaseConfig:  
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        values = [name for name in dir(self) if not name.startswith('__')]
        params = {name: self.__getattribute__(name) for name in values}
        print_params = json.dumps(params, indent=4)
        string = f'{type(self).__name__}:\n{print_params}'
        return string

class ImagesConfig(BaseConfig):
    in_channels: int = 6
    num_classes: int = 2
    img_resolution: int = (128, 128)

class BoxConfig(BaseConfig):
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

class BackboneConfig(BaseConfig):
    backbone_name: str = "resnet50"
    num_feature_maps: int = 5
    num_out_channels: int = 256

# TODO FIX
class RPNConfig(BaseConfig):
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_score_thresh: float = 0.0

# TODO FIX
class BoxConfig(BaseConfig):
    # Box parameters
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    bbox_reg_weights: Optional[Tuple[float, float, float, float]] = None

# TODO FIX
class EllipseConfig(BaseConfig):
    pass

# TODO FIX
class MetaConfig():
    def __init__(
        self, 
        images = ImagesConfig, 
        backbone = BackboneConfig, 
        rpn = RPNConfig, 
        box = BoxConfig, 
        ellipse = EllipseConfig
    ):
        self.images = images()
        self.backbone = backbone()
        self.rpn = rpn()
        self.ellipse = ellipse()
    
    def __repr__(self):
        return f'{self.images}\n\n{self.backbone}\n\n{self.rpn}\n\n{self.ellipse}'
    
    def to_json(self, path):
        
        dictionary  = {key: self.__dict__[key].__dic for key in self.__dict__}
        json_object = json.dumps(, indent=4)
        with open(path, "w") as f:
            f.write(json_object)
        
    def from_json(self):
        pass
    # @classmethod
    # def from_dict(cls, the_dict) -> PlotConfig:
    #     """Create a PlotConfig object from a dict"""
    #     return cls(
    #         plotname=the_dict["plotname"],
    #         classname=the_dict["classname"],
    #         plot_config=the_dict["plot_config"],
    #     )
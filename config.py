"""Config setup architecture"""
import json


class BaseConfig:  
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def _to_dict(self):
        values = [name for name in dir(self) if not name.startswith('_')]
        return {name: self.__getattribute__(name) for name in values}

    def __repr__(self):
        # TODO Better representation of config (?)
        print_params = json.dumps(self._to_dict(), indent=4)
        string = f'{type(self).__name__}:\n{print_params}'
        return string


class ImagesConfig(BaseConfig):
    in_channels = 5
    num_classes = 2  # galaxy + background
    img_resolution = (128, 128)


class BackboneConfig(BaseConfig):
    name = "resnet50"
    input_channels = 6


class RPNConfig(BaseConfig):
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    pre_nms_top_n_train = 2000
    pre_nms_top_n_test = 1000
    post_nms_top_n_train = 2000
    post_nms_top_n_test = 1000
    nms_thresh = 0.7
    fg_iou_thresh = 0.7
    bg_iou_thresh = 0.3
    batch_size_per_image = 128
    positive_fraction = 0.5
    score_thresh = 0.0


class ROIBoxHeadConfig(BaseConfig):
    output_pool_size = 7
    score_thresh = 0.05
    nms_thresh = 0.5
    detections_per_img = 100
    fg_iou_thresh = 0.5
    bg_iou_thresh = 0.5
    batch_size_per_image = 512
    positive_fraction = 0.25
    bbox_reg_weights = None

# TODO FIX
class EllipseConfig(BaseConfig):
    pass

# # TODO FIX
# class MetaConfig():
#     def __init__(
#         self,
#         images = ImagesConfig,
#         backbone = BackboneConfig,
#         rpn = RPNConfig,
#         box = BoxConfig,
#         ellipse = EllipseConfig
#     ):
#         self.images = images()
#         self.backbone = backbone()
#         self.rpn = rpn()
#         self.ellipse = ellipse()
#
#     def __repr__(self):
#         return f'{self.images}\n\n{self.backbone}\n\n{self.rpn}\n\n{self.ellipse}'
#
#     def to_json(self, path):
#
#         dictionary  = {key: self.__dict__[key].__dic for key in self.__dict__}
#         #json_object = json.dumps(, indent=4)
#         with open(path, "w") as f:
#             f.write(json_object)
#
#     def from_json(self):
#         pass
#     # @classmethod
#     # def from_dict(cls, the_dict) -> PlotConfig:
#     #     """Create a PlotConfig object from a dict"""
#     #     return cls(
#     #         plotname=the_dict["plotname"],
#     #         classname=the_dict["classname"],
#     #         plot_config=the_dict["plot_config"],
#     #     )

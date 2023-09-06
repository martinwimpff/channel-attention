from channel_attention.models.eegnet import EEGNet
from channel_attention.models.shallownet import ShallowNet


model_dict = dict(
    EEGNet=EEGNet,
    ShallowNet=ShallowNet
)


def get_model_cls(model_name):
    return model_dict[model_name]

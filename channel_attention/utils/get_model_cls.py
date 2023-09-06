from channel_attention.models import ATCNet
from channel_attention.models import BaseNet
from channel_attention.models import EEGNet
from channel_attention.models import ShallowNet


model_dict = dict(
    ATCNet=ATCNet,
    BaseNet=BaseNet,
    EEGNet=EEGNet,
    ShallowNet=ShallowNet
)


def get_model_cls(model_name):
    return model_dict[model_name]

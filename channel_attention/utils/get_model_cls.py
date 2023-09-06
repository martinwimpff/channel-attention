from channel_attention.models import (
    ATCNet,
    BaseNet,
    EEGConformer,
    EEGNet,
    ShallowNet,
    TSSEFFNet
)


model_dict = dict(
    ATCNet=ATCNet,
    BaseNet=BaseNet,
    EEGConformer=EEGConformer,
    EEGNet=EEGNet,
    ShallowNet=ShallowNet,
    TSSEFFNet=TSSEFFNet
)


def get_model_cls(model_name):
    return model_dict[model_name]

from channel_attention.models.eegnet import EEGNet


model_dict = dict(
    EEGNet=EEGNet
)


def get_model_cls(model_name):
    return model_dict[model_name]

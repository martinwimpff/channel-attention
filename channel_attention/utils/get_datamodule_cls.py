from channel_attention.datamodules.bcic import BCICIV2a, BCICIV2b


def get_datamodule_cls(dataset_name):
    if dataset_name == "bcic2a":
        datamodule_cls = BCICIV2a
    elif dataset_name == "bcic2b":
        datamodule_cls = BCICIV2b
    else:
        raise NotImplementedError(f"No dataset with name: {dataset_name}")

    return datamodule_cls

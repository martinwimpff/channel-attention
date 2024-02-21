from channel_attention.datamodules import BCICIII_IVa, BCICIII_IVaLOSO, BCICIV2a, \
    BCICIV2b, HighGamma


def get_datamodule_cls(dataset_name):
    if dataset_name == "bcic3":
        datamodule_cls = BCICIII_IVa
    elif dataset_name == "bcic3_loso":
        datamodule_cls = BCICIII_IVaLOSO
    elif dataset_name == "bcic2a":
        datamodule_cls = BCICIV2a
    elif dataset_name == "bcic2b":
        datamodule_cls = BCICIV2b
    elif dataset_name == "hgd":
        datamodule_cls = HighGamma
    else:
        raise NotImplementedError(f"No dataset with name: {dataset_name}")

    return datamodule_cls

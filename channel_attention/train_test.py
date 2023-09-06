import numpy as np
from pytorch_lightning import Trainer

from channel_attention.utils.get_datamodule_cls import get_datamodule_cls
from channel_attention.utils.get_model_cls import get_model_cls
from channel_attention.utils.seed import seed_everything


def train_and_test(config):
    # get datamodule_cls and model_cls
    model_cls = get_model_cls(model_name=config["model"])
    datamodule_cls = get_datamodule_cls(dataset_name=config["dataset_name"])

    if config["subject_ids"] == "all":
        subject_ids = datamodule_cls.all_subject_ids
    else:
        subject_ids = [config["subject_ids"]]

    test_accs = []
    for subject_id in subject_ids:
        seed_everything(config["seed"])

        # set up the trainer
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy="auto",
            logger=False,
            enable_checkpointing=False
        )

        # set up the datamodule
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)

        # train and test
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])
        trainer.fit(model, datamodule=datamodule)
        test_results = trainer.test(model, datamodule)
        test_accs.append(test_results[0]["test_acc"])

    # print overall acc
    print(f"test_acc: {np.mean(test_accs)}")

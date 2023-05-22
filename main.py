from kie.trainer import TrainConfig, TrainConfig, Trainer
import icecream

icecream.install()
if __name__ == "__main__":
    import torch
    from icecream import install
    from transformers import AutoTokenizer
    from kie.configs import ModelConfig, TrainConfig
    train_config = TrainConfig.from_file("configs/training.yaml")
    model_config = ModelConfig.from_file("configs/kie-invoice.yaml")

    install()
    trainer = Trainer(train_config, model_config)
    trainer.train()

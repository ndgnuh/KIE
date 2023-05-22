def dev(args):
    import icecream
    icecream.install()
    import torch
    from icecream import install
    from transformers import AutoTokenizer
    from kie.configs import ModelConfig, TrainConfig
    from kie.data import KieDataset
    from kie.predictor import Predictor
    from kie.prettyprint import simple_postprocess as prettyprint

    train_config = TrainConfig.from_file("configs/training.yaml")
    model_config = ModelConfig.from_file("configs/kie-invoice.yaml")

    dataset = KieDataset("data/val.json")
    sample = dataset[0]
    install()

    ctx = Predictor(model_config)
    output = ctx.predict(sample=sample)
    output = ctx.predict(
        texts=sample.texts,
        boxes=sample.boxes,
        image_width=sample.image_width,
        image_height=sample.image_height,
    )
    ic(ctx.pretty_format(output))


def train(args):
    from kie.trainer import Trainer
    from kie.configs import ModelConfig, TrainConfig

    train_config = TrainConfig.from_file(args.train_config)
    model_config = ModelConfig.from_file(args.model_config)
    trainer = Trainer(train_config, model_config)
    trainer.train()


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    subparser = parser.add_subparsers(dest="action", required=True)

    ##
    ## Training
    ##
    train_parser = subparser.add_parser("train")
    train_parser.add_argument(
        "--model", "-m", dest="model_config", help="Model configuration file"
    )
    train_parser.add_argument(
        "--experiment", "-e", dest="train_config", help="Training configuration file"
    )

    ##
    ## Dev
    ##
    _ = subparser.add_parser("dev")
    args = parser.parse_args()
    if args.action == "train":
        train(args)
    elif args.action == "dev":
        dev(args)


if __name__ == "__main__":
    import os

    main()

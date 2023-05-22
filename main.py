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


def test(args):
    from pprint import pprint
    from collections import defaultdict
    from kie.data import KieDataset
    from kie.predictor import Predictor
    from kie.configs import ModelConfig
    from tqdm import tqdm
    from kie.metrics import (
        Metric,
        Statistics,
        get_tensor_f1,
        get_e2e_f1,
        get_e2e_f1_per_class,
    )
    import numpy as np

    # Trainer config with dummies
    metrics = defaultdict(lambda: Statistics(np.mean))
    metrics_by_fields = defaultdict(lambda: Statistics(np.mean))
    model_config = ModelConfig.from_file(args.model_config)
    predictor = Predictor(model_config)
    dataset = KieDataset(args.data)
    for sample in tqdm(dataset, "Testing", dynamic_ncols=True):
        pr = predictor.predict_sample(sample)
        pr_raw = predictor.processor.encode(pr)
        gt_raw = predictor.processor.encode(sample)
        metrics['f1_raw_links'].append(get_tensor_f1(pr_raw.classes, gt_raw.classes))
        metrics['f1_raw_relations'].append(get_tensor_f1(pr_raw.adj, gt_raw.adj))

        pr = predictor.pretty_format(pr)
        gt = predictor.pretty_format(sample)
        metrics['f1_e2e'].append(get_e2e_f1(pr, gt))
        e2e_f1s = get_e2e_f1_per_class(pr, gt, dataset.classes)
        for c in dataset.classes:
            metrics_by_fields[c].append(e2e_f1s[c])

    metrics = {k: v.get() for k, v in metrics.items()}
    metrics['by_clases'] = {k: v.get() for k, v in metrics_by_fields.items()}
    pprint(metrics)


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

    #
    # Testing
    #
    test_parser = subparser.add_parser("test")
    test_parser.add_argument(
        "--model", "-m", dest="model_config", help="Model configuration file"
    )
    test_parser.add_argument("--data", "-d", dest="data", help="Dataset file")
    args = parser.parse_args()
    if args.action == "train":
        train(args)
    elif args.action == "dev":
        dev(args)
    elif args.action == "test":
        test(args)


if __name__ == "__main__":
    import os

    main()

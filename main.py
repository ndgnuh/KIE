from kie.trainer import TrainConfig, KieConfig, TrainConfig, Trainer
if __name__ == "__main__":
    import torch
    from icecream import install
    from transformers import AutoTokenizer

    install()
    tokenizer_name = "vinai/phobert-base"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # dataloader = make_dataloader("data/inv_aug_noref_noimg.json", prepare_fn(tokenizer))

    model_config = KieConfig(
        backbone_name="microsoft/layoutlm-base-cased",
        word_embedding_name=tokenizer_name,
        head_dims=256,
        num_classes=15,
    )
    train_config = TrainConfig(
        total_steps=1000,
        validate_every=100,
        train_data="data/inv_aug_noref_noimg.json",
        validate_data="data/inv_aug_noref_noimg.json",
        lr=8e-5,
    )
    trainer = Trainer(train_config, model_config)
    model = trainer.model.eval()
    model.load_state_dict(torch.load("model.pt"))

    batch = next(iter(trainer.train_loader))
    with torch.no_grad():
        output = model(batch)
        ic(output.loss)
        ic(output.class_logits)
        ic(output.relation_logits)

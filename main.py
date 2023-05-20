from kie.trainer import TrainConfig, KieConfig, TrainConfig, Trainer
import icecream

icecream.install()
if __name__ == "__main__":
    import torch
    from icecream import install
    from transformers import AutoTokenizer

    install()
    tokenizer_name = "vinai/phobert-base"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # dataloader = make_dataloader("data/inv_aug_noref_noimg.json", prepare_fn(tokenizer))

    model_config = KieConfig(
        backbone_name="naver-clova-ocr/bros-base-uncased",
        word_embedding_name=tokenizer_name,
        head_dims=256,
        num_classes=17,
        classes=[
            "barcode",
            "date",
            "customer",
            "agency",
            "representative",
            "address",
            "id",
            "booknum",
            "declnum",
            "unit",
            "quantity",
            "tweight",
            "kweight",
            "signname"
        ],
    )
    train_config = TrainConfig(
        total_steps=10000,
        validate_every=200,
        train_data="data/val.json",
        validate_data="data/val.json",
        dataloader=dict(
            batch_size=1,
            num_workers=2,
        ),
        lr=5e-5,
    )
    trainer = Trainer(train_config, model_config)
    # trainer.model.load_state_dict(torch.load("model.pt"), strict=False)
    trainer.train()
    # model = trainer.model.eval()

    # batch = next(iter(trainer.train_loader))
    # with torch.no_grad():
    #     output = model(batch)
    #     ic(output.loss)
    #     ic(output.class_logits)
    #     ic(output.relation_logits)

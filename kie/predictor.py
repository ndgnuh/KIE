import torch
from .configs import ModelConfig
from .models import KieModel, Tokenizer
from .processor_v2 import Processor
from .data.schema import Sample
from .prettyprint import simple_postprocess as prettyprint


class Predictor:
    def __init__(self, model_config):
        self.model = KieModel(model_config)
        self.tokenizer = Tokenizer(model_config)
        if model_config.inference_weights is not None:
            self.model.load_state_dict(
                torch.load(model_config.inference_weights, map_location="cpu")
            )
        self.processor = Processor(
            tokenizer=self.tokenizer, classes=model_config.classes
        )
        self.model_config = model_config

    def predict(self, sample=None, texts=None, boxes=None, image_width=None, image_height=None):
        # Because @singledispatchmethod sucks, that's why
        if sample is not None:
            return self.predict_sample(sample)
        else:
            return self.predict_texts_boxes(texts, boxes, image_width, image_height)
        raise NotImplementedError()

    def predict_sample(self, sample: Sample) -> Sample:
        encoded = self.processor.encode(sample).to_tensor().to_batch()
        outputs = self.model(encoded)
        encoded.adj = outputs.relations
        encoded.classes = outputs.classes
        decoded = self.processor.decode(encoded.to_numpy()[0])
        return decoded

    def predict_texts_boxes(self, texts: list, boxes: list, image_width: int, image_height: int) -> Sample:
        sample = Sample(texts=texts, boxes=boxes, classes={}, links=[],
                        image_width=image_width,
                        image_height=image_height)
        return self.predict(sample=sample)

    def pretty_format(self, sample: Sample):
        return prettyprint(sample, self.model_config.classes)

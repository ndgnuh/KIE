from pydantic import BaseModel, Field
from typing import Callable, List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer
from kie.fileio import read


Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]


class Sample(BaseModel):
    texts: List[str]
    boxes: List[Polygon]
    image_width: int
    image_height: int
    image_base64: Optional[str]
    classes: Dict[int, int] = Field(default_factory=dict)


def idendity(x):
    return x


def prepare_input(tokenizer, sample: Sample):
    texts = sample.texts
    boxes = np.array(sample.texts)


class KieDataset(Dataset):
    def __init__(self, root, transform=idendity):
        super().__init__()
        data = read(root)
        self.root = root
        self.transform = transform
        self.classes = data["classes"]
        self.samples = [Sample.parse_obj(sample) for sample in data["samples"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample: Sample = self.samples[idx]
        sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    dataset = KieDataset("./data/inv_aug_noref_noimg.json")
    dl = DataLoader(dataset, batch_size=4)
    print(dataset)
    dataset[0]
    next(dl)

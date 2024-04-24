from torch.utils.data import DataLoader

from .textile_dataset import TextileDatasetStage1, TextileDatasetStage2


DATASET_LIST = {
    'stage1': TextileDatasetStage1,
    'stage2': TextileDatasetStage2,
}

def load_dataset(configs, stage, split):
    dataset = DATASET_LIST[stage](**configs, split=split)
    dataloader = DataLoader(dataset, batch_size=configs['batch_size'] if split == 'train' else 1,
                            shuffle=split == 'train', num_workers=configs['num_workers'], pin_memory=True)
    return dataloader



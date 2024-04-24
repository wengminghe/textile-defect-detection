import argparse
from omegaconf import OmegaConf

from trainers import Trainer, TrainerStage1, TrainerStage2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lace.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    if config['stage'] == 0:
        trainer = Trainer(config)
    elif config['stage'] == 1:
        trainer = TrainerStage1(config)
    else:
        trainer = TrainerStage2(config)
    trainer.run()


if __name__ == '__main__':
    main()


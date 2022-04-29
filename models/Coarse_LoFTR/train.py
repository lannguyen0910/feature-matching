import argparse
from trainer.trainer import Trainer
from configs.config import BaseConfig


def main():
    parser = argparse.ArgumentParser(description='LoFTR knowledge distillation.')
    parser.add_argument('--path', type=str, default='/data_sets/BlendedMVS',
                        help='Path to the dataset.')
    parser.add_argument('--checkpoint_path', type=str,
                        default='weights',
                        help='Where to store a log information and checkpoints.')
    parser.add_argument('--weights', type=str, default='weights/outdoor_ds.ckpt',
                        help='Path to the LoFTR teacher network weights.')

    opt = parser.parse_args()
    print(opt)

    config = BaseConfig()
    trainer = Trainer(config, opt.weights, opt.path, opt.checkpoint_path)
    trainer.train('LoFTR')


if __name__ == '__main__':
    main()

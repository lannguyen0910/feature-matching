from configs.data.base import cfg


cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

cfg.DATASET.DEPTH0_PATH = "/content/data/depth_maps/"
cfg.DATASET.DEPTH1_PATH = "/content/data/depth_maps/"

cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val

# 368 scenes in total for MegaDepth
# (with difficulty balanced (further split each scene to 3 sub-scenes))
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.MGDPT_IMG_RESIZE = 840  # for training on 32GB meme GPUs

from src.config.default import get_cfg_defaults
config = get_cfg_defaults()

# INDOOT lofrt_ds_quadtree config
config.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
config.LOFTR.MATCH_COARSE.SPARSE_SPVS = False
config.LOFTR.RESNETFPN.INITIAL_DIM = 128
config.LOFTR.RESNETFPN.BLOCK_DIMS=[128, 196, 256]
config.LOFTR.COARSE.D_MODEL = 256
config.LOFTR.COARSE.BLOCK_TYPE = 'quadtree'
config.LOFTR.COARSE.ATTN_TYPE = 'B'
config.LOFTR.COARSE.TOPKS=[32, 16, 16]
config.LOFTR.FINE.D_MODEL = 128
config.TRAINER.WORLD_SIZE = 1 # 8
config.TRAINER.CANONICAL_BS = 32
config.TRAINER.TRUE_BATCH_SIZE = 1
_scaling = 1
config.TRAINER.ENABLE_PLOTTING = False
config.TRAINER.SCALING = _scaling
config.TRAINER.TRUE_LR = 1e-3 # 1e-4 config.TRAINER.CANONICAL_LR * _scaling
config.TRAINER.WARMUP_STEP = 0 #math.floor(config.TRAINER.WARMUP_STEP / _scaling)
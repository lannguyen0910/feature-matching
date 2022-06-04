import copy

default_cfg = {'backbone_type': 'ResNetFPN',
               'resolution': (8, 2),
               'fine_window_size': 5,
               'fine_concat_coarse_feat': True,
               'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
               'coarse': {'d_model': 256,
                          'd_ffn': 256,
                          'nhead': 8,
                          'layer_names': ['self',
                                          'cross',
                                          'self',
                                          'cross',
                                          'self',
                                          'cross',
                                          'self',
                                          'cross'],
                          'attention': 'linear',
                          'temp_bug_fix': False},
               'match_coarse': {'thr': 0.2,
                                'border_rm': 2,
                                'match_type': 'dual_softmax',
                                'dsmax_temperature': 0.1,
                                'skh_iters': 3,
                                'skh_init_bin_score': 1.0,
                                'skh_prefilter': True,
                                'train_coarse_percent': 0.4,
                                'train_pad_num_gt_min': 200},
               'fine': {'d_model': 128,
                        'd_ffn': 128,
                        'nhead': 8,
                        'layer_names': ['self', 'cross'],
                        'attention': 'linear'}}

# dual softmax (ds)
ds_cfg = copy.deepcopy(default_cfg)
ds_cfg['match_coarse']['thr'] = 0.25
# sinkhorn (ot)
ot_cfg = copy.deepcopy(default_cfg)
ot_cfg['match_coarse']['thr'] = 0.325
ot_cfg['match_coarse']['match_type'] = 'sinkhorn'
ot_cfg['match_coarse']['sparse_spvs'] = False
ot_cfg['match_coarse']['skh_iters'] = 4

class BaseConfig:
    def __init__(self):
        self.cuda = True

        self.batch_size = 32
        self.batch_size_divider = 8  # Used for gradient accumulation
        self.use_amp = True
        self.epochs = 50
        self.epoch_size = 0
        # used for identification of correspondent points in dataset image pairs
        self.depth_tolerance = 0.005

        self.learning_rate = 0.01
        self.scheduler_step_size = 15  # epochs
        self.scheduler_gamma = 0.01

        self.with_teacher = True
        self.student_coeff = 0.3
        self.distillation_coeff = 1. - self.student_coeff
        self.temperature = 5.0
        # distillation loss is usually too small - make it bigger
        self.distill_ampl_coeff = 10

        self.amp_scale = 2.**8
        self.data_loader_num_workers = 4
        self.write_statistics = True
        self.statistics_period = 10

        self.image_size = (640, 480)

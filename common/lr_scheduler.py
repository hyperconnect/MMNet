def factory(args, log, global_step_from_checkpoint=None, global_step=None, dataset=None):
    learning_rate_scheduler = FixedLR(args, log)
    return learning_rate_scheduler


def add_arguments(parser):
    FixedLR.add_arguments(parser)


class FixedLR:
    def __init__(
        self, args, logger
    ):
        self.args = args
        self.logger = logger
        assert hasattr(self.args, "learning_rate") and isinstance(self.args.learning_rate, float)
        self.learning_rate = self.args.learning_rate

        self.placeholder = self.learning_rate
        self.should_feed_dict = False

    @staticmethod
    def add_arguments(parser):
        g_lr = parser.add_argument_group("Learning Rate Arguments")
        g_lr.add_argument("--learning_rate", default=1e-4, type=float, help="Initial learning rate for gradient update")

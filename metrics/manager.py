import metrics.ops as mops
import metrics.parser as parser
from metrics.base import MetricManagerBase
from metrics.summaries import Summaries


class MattingMetricManager(MetricManagerBase):
    _metric_input_data_parser = parser.MattingDataParser

    def __init__(self,
                 is_training: bool,
                 save_evaluation_image: bool,
                 exclude_metric_names: list,
                 summary: Summaries):
        super().__init__(exclude_metric_names, summary)
        self.register_metrics([
            # misc
            mops.InferenceTimeMetricOp(),
            # tensor ops
            mops.LossesMetricOp(),
            mops.ImageSummaryOp(),

            mops.MADMetricOp(),
            mops.GaussianGradMetricOp(),
        ])

        if not is_training and save_evaluation_image:
            self.register_metrics([
                mops.MiscImageRetrieveOp(),
                mops.MiscImageSaveOp(),
            ])

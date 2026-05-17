import numpy as np
from recbole.evaluator import AbstractMetric, DataStruct
from recbole.utils import EvaluatorType


class CustomMetric(AbstractMetric):
    """
    TODO: Please follow https://recbole.io/docs/developer_guide/customize_metrics.html
    to create customized metrics.
    """
    metric_need = []
    metric_type = EvaluatorType.RANKING
    smaller = False

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pass

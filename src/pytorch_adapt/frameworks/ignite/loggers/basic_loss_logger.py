from pytorch_metric_learning.utils import common_functions as pml_cf

from .ignite_empty_logger import IgniteEmptyLogger


class BasicLossLogger(IgniteEmptyLogger):
    def __init__(self):
        from record_keeper import RecordKeeper, RecordWriter
        from record_keeper import utils as rk_utils

        # Define in here so that record_keeper import stays in init
        class LossKeeper(RecordWriter):
            def __init__(self):
                self.records = self.get_empty_nested_dict()

            def append(self, group_name, series_name, input_val, iteration):
                curr_dict = self.records[group_name]
                append_this = rk_utils.convert_to_scalar(input_val)
                if series_name not in curr_dict:
                    curr_dict[series_name] = []
                curr_dict[series_name].append(append_this)

        self.record_keeper = RecordKeeper(
            record_writer=LossKeeper(),
            attributes_to_search_for=pml_cf.list_of_recordable_attributes_list_names(),
        )

    def add_training(self, _):
        def fn(engine):
            self.record_keeper.update_records(
                engine.state.output, engine.state.iteration
            )

        return fn

    def get_losses(self):
        rw = self.record_keeper.record_writer
        output = rw.records
        rw.records = rw.get_empty_nested_dict()
        return output

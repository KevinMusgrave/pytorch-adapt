from pytorch_metric_learning.utils import common_functions as pml_cf

from .ignite_empty_logger import IgniteEmptyLogger


class BasicLossLogger(IgniteEmptyLogger):
    def __init__(self, folder):
        from record_keeper import RecordKeeper, RecordWriter

        self.record_keeper = RecordKeeper(
            record_writer=RecordWriter(folder),
            attributes_to_search_for=pml_cf.list_of_recordable_attributes_list_names(),
        )

    def add_training(self, adapter):
        def fn(engine):
            self.record_keeper.update_records(
                {"engine_output": engine.state.output}, engine.state.iteration
            )

        return fn

    def get_losses(self):
        rw = self.record_keeper.record_writer
        output = rw.records
        rw.records = rw.get_empty_nested_dict()

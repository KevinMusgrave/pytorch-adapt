class IgniteEmptyLogger:
    def add_training(self, *args, **kwargs):
        def fn(*args, **kwargs):
            pass

        return fn

    def add_validation(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

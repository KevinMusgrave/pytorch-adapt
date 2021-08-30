class ForwardOnlyValidator:
    def run(self, adapter, **kwargs):
        if "validator" not in kwargs:
            raise KeyError(
                "An adapter validator is required when using ForwardOnlyValidator"
            )
        return adapter.run(**kwargs)

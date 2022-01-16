from io import StringIO

HOOK_STREAM = StringIO()


class HookLogger:
    def __init__(self, name):
        self.name = name

    def __call__(self, x):
        HOOK_STREAM.write(f"{self.name}: {x}\n")


def reset_hook_logger():
    HOOK_STREAM.seek(0)
    HOOK_STREAM.truncate(0)

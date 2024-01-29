from functools import wraps


def _install_with_pip(file=None, libraries=None):
    import os
    import subprocess
    import sys

    _libraries = {}
    if file is not None:
        with open(file, "r") as reqs:
            lines = [line.split("\n")[0] for line in reqs.readlines()]
            for line in lines:
                result = line.split("==")
                if len(result) == 2:
                    library, version = result[0], result[1]
                    _libraries[library] = version
                elif len(result) == 1:
                    library = result[0]
                    _libraries[library] = ""
                else:
                    raise ValueError("Each line in requirements.txt file ")

    else:
        _libraries = libraries

    for library, version in _libraries.items():
        if version != "":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    library + "==" + version,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", library])


def pip(file=None, packages=None):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            _install_with_pip(file=file, libraries=packages)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def enable_decorator(dec, flag):
    def decorator(func):
        if flag:
            return dec(func)
        return func
    return decorator
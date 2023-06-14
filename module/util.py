import os

def try_make_dir(dirname: str):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

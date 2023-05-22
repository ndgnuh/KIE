import gdown


def download(url, output=None, force=False):
    if force:
        return gdown.download(url, output)
    else:
        return gdown.cached_download(url, output)


def down_or_load(path):
    schemes = ["https://", "http://"]
    for scheme in schemes:
        if path.startswith(scheme):
            return download(path)
    return path

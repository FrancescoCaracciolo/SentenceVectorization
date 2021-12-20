import requests

MODELS = {
    "glove.42B.300d": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip",
    "glove.840B.300d": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip",
    "glove.6B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    "glove.twitter.27B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip",
}


def download_model(model_name: str, filepath: str) -> bool:
    if model_name not in MODELS:
        return False
    else:
        return __download_file(filepath, MODELS[model_name])


def __download_file(filename: str, url: str, status: bool = False) -> bool:
    """Downloads files, private function used in download_model

    Args:
        filename (str): path of the file where to download
        url (str): url of the source
        debug (bool): Show download logging

    Returns:
        bool: Returns True if the download was successful, false if not
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if (status):
                        print("e")
                    f.write(chunk)
    except Exception as e:
        if status:
            print("Error while downloading {filename}: " + str(e))
        return False
    else:
        return True

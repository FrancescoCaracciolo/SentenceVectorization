import requests
import os
import zipfile
from tqdm import tqdm
from SentenceVectorization.vectors import SentVE

MODELS = {
    "glove.42B.300d": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip",
    "glove.840B.300d": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip",
    "glove.6B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    "glove.twitter.27B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip",
}
SV = SentVE()


def download_model(model_name: str, directory: str, status: bool = True) -> bool:
    """Downloads GloVE pre-trained models in a specified folder

    Args:
        model_name (str): Model name, choose from MODELS
        directory (str): Directory where the models are saved
        status (bool, optional): Enable or disable logging. Defaults to True.

    Returns:
        bool: True on success, False on failiture
    """
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except Exception as e:  # If cannot create directory
            print("Cannot create directory" + str(e)) if status else False
        return False

    filepath = os.path.join(directory, model_name)
    if model_name not in MODELS:
        return False
    else:
        print("Downloading the file...")
        c1 = __download_file(filepath + ".zip", MODELS[model_name], True)
        if c1:
            print("Extracting the model...")
            c2 = __unzip(filepath + ".zip", directory)
        os.remove(filepath + ".zip")
        return c1 and c2


def __download_file(filename: str, url: str, status: bool = False) -> bool:
    """Downloads files, private function used in download_model

    Args:
        filename (str): path of the file where to download
        url (str): url of the source
        debug (bool): Show download logging

    Returns:
        bool: Returns True if the download was successful, false if not
    """
    downloaded = 0  # Byte downloaded, used for logging
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            filesize = int(r.headers['Content-length'])
            pbar = tqdm(total=int(filesize)) if status else False  # Download bar
            with open(filename, 'wb') as f:  # Download the file
                for chunk in r.iter_content(chunk_size=8192):
                    downloaded += 8192
                    pbar.update(8192) if status else False
                    f.write(chunk)
            pbar.close if status else False
            print(downloaded, filesize)
    except Exception as e:
        if status:
            print("Error while downloading {filename}: " + str(e))
        return False
    else:
        return True


def __unzip(filename: str, directory: str, status: bool = False) -> bool:
    try:
        zip = zipfile.ZipFile(filename)
        if status:
            print("List of models: ")
        for x in zip.filelist:
            print(x.filename)
        zip.extractall(directory)
    except Exception as e:
        if status:
            print("Error while extracting {filename}: " + str(e))
            return False
    else:
        return True

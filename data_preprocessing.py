from pathlib import Path
import tarfile
import urllib.request

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

def download_and_extract(data_dir="data"):
    """Download the TensorFlow flower_photos dataset once and extract it."""
    data_path = Path(data_dir)
    flower_dir = data_path / "flower_photos"
    if flower_dir.exists() and any(p.is_dir() for p in flower_dir.iterdir()):
        return flower_dir

    data_path.mkdir(parents=True, exist_ok=True)
    archive = data_path / "flower_photos.tgz"
    print("Downloading flower dataset...")
    urllib.request.urlretrieve(DATA_URL, archive)
    print("Extracting dataset...")
    with tarfile.open(archive, "r:gz") as tar:
        # Python 3.10/3.11 do not support the newer filter argument.
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=data_path, filter="data")
        else:
            tar.extractall(path=data_path)
    return flower_dir

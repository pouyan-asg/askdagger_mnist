import os
import gdown


if __name__ == "__main__":
    ids = [
        "1UhF13-qpdfFpXOlIPGdpeuwLAWfV98bt",
        "1-pBiUDs3jmx1hV80obPptZdoyBFybkIn",
        "1HYqBYuwPfHmzS_NoSLQGIE6BrF_ZJJiK",
        "1f5CnucnTgGZkiSKs7-S2f18grWMIdoIC",
    ]
    outputs = [
        "results_sensitivity.zip",
        "results_specificity.zip",
        "results_success.zip",
        "results.zip",
    ]
    for id, output in zip(ids, outputs):
        url = f"https://drive.google.com/uc?id={id}"
        gdown.download(url, output, quiet=False)
        os.system(f"unzip -o {output}")
        os.system(f"rm {output}")
        print(f"Downloaded {output}")

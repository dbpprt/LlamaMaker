import argparse

import requests


DATASET = {
    "": "",
}


def download_file(url: str, filename: str):
    print(f"Downloading {url}...")
    resp = requests.get(url)
    with open(filename, "wb") as f:
        f.write(resp.content)


def main():
    parser = argparse.ArgumentParser("Mintaka dataset downloader", usage="acquire [<args>]", allow_abbrev=False)

    parser.add_argument(
        "--target_directory",
        type=str,
        help="The target directory to download the dataset into.",
        default="./data/mintaka",
    )

    args = parser.parse_args()
    print(f"Downloading the mintaka dataset to {args.target_directory}.")


if __name__ == "__main__":
    main()

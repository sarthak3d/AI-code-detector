#!/usr/bin/env python
"""
Usage:
    download_dataset.py DESTINATION_DIR --set SET_NAME

Options:
    -h --help   Show this screen.
    -s --set    Chose set to download (function/class/inline/all)
"""

import os, requests, zipfile
from subprocess import call
from argparse import ArgumentParser
from tqdm import tqdm

def get_args():
    parser = ArgumentParser(description='merge dataset')
    parser.add_argument(
        "DESTINATION_DIR",
        type=str,
        help="Destination dir",
    )
    parser.add_argument(
        "--set",
        "-s",
        default='function',
        type=str,
        help="Chose set to download (function/class/inline/all)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    destination_dir = os.path.abspath(args.DESTINATION_DIR)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    os.chdir(destination_dir)

    if args.set == 'all':
        set_name = ['function', 'class', 'inline']
    else:
        set_name = [args.set]
        
    for language in ('python', 'c_sharp', 'c', 'rust'):#('javascript', 'java', 'ruby', 'php', 'go', 'cpp', 'python', 'c_sharp', 'c', 'rust'):
        for _name in set_name:
            url = f"https://ai4code.blob.core.windows.net/thevault/v1/{_name}/{language}.zip"
            output_path = os.path.join(destination_dir, f"{language}.zip")

            print(f"Downloading {url} ...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            if response.status_code == 200:
                with open(output_path, "wb") as f, tqdm(
                    desc=f"{language}.zip",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        progress_bar.update(size)
                print(f"Saved to {output_path}")
            else:
                print(f"Failed to download {url} (status {response.status_code})")
                continue

            extract_path = os.path.join(destination_dir, language)

            with open(output_path, "rb") as f:
                signature = f.read(2)
            if signature != b"PK":
                print(f"{output_path} is not a valid ZIP file, skipping...")
                continue

            print(f"Extracting {language}.zip -> {language}")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                # Calculate total uncompressed size for byte-based progress
                total_size = sum(info.file_size for info in zip_ref.infolist())
                
                with tqdm(
                    total=total_size,
                    desc=f"Extracting {language}",
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress_bar:
                    for member in zip_ref.infolist():
                        # Skip directories
                        if member.is_dir():
                            zip_ref.extract(member, extract_path)
                            continue
                        
                        # Extract file with progress tracking
                        target_path = os.path.join(extract_path, member.filename)
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        
                        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                            while True:
                                chunk = source.read(8192)
                                if not chunk:
                                    break
                                target.write(chunk)
                                progress_bar.update(len(chunk))
            print(f"Extracted {language} Complete.")
            os.remove(output_path)
    print("All datasets downloaded and extracted successfully.")

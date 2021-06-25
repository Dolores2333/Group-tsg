"""
Datasets module
"""
import logging
import ssl

from fire import Fire

from datasets.tourism import TourismDataset


def build():
    """
    Download all datasets.
    """
    # Fix for: Hostname mismatch, certificate is not valid for 'mcompetitions.unic.ac.cy'
    ssl._create_default_https_context = ssl._create_unverified_context

    logging.info('\n\nTourism Dataset')
    TourismDataset.download()

if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()

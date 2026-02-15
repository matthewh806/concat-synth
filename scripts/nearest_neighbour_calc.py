from common import setup_corpus, get_parser
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]

'''
This script loads all of the files in the audio_downloads directory, analyses each AudioSnippet
and performs a nearest neighbour search using a random AudioSnippet from the list
'''

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    corpus, config = setup_corpus(config_path=args.config)
    target = corpus.get_random_snippet()

    logging.info(f"Finding nearest neighbour for target: {target}")
    
    nearest_neighbour = corpus.find_best_neighbour(
        target_snippet=target,
        exclusion_list=[]
    )

    if nearest_neighbour:
        logging.info(f"Found Nearest neighbour: {nearest_neighbour}")
from concurrent.futures import ThreadPoolExecutor, as_completed

def collect_snippets_parallel(
        backend,
        queries,
        max_snippets = 32,
        max_workers = 4
):
    '''
    Parallelised task manager for downloading / retrieving audio. 
    Uses a ThreadPoolExector to separate the download tasks into different threads
    
    :param backend: The source of the audio files to be downloaded (e.g. FreesoundAudioDownloader, YoutubeAudioDownloader)
    :param queries: List of the query strings to download
    :param max_snippets: The maximum number of downloads to perform
    :param max_workers: The number of workers to parallelise this task with

    :return List of paths to the audio files which have been downloaded
    '''
    
    snippets = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(backend.get_snippets, query): query
            for query in queries
        }

        for future in as_completed(futures):
            query = futures[future]
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                print(f"Download ({completed} / {max_snippets}) failed for {query}: {e}")
                continue

            print(f"Download ({completed} / {max_snippets}) completed for {query}")
            for snippet in result:
                snippets.append(snippet)
                if len(snippets) >= max_snippets:
                       return snippets

    return snippets

from concurrent.futures import ThreadPoolExecutor, as_completed

def collect_snippets_parallel(
        backend,
        queries,
        max_snippets = 32,
        max_workers = 4
):
    
    snippets = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(lambda q=query: list(backend.get_snippets(q))): query
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

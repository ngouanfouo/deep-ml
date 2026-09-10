from collections import deque

def crawl(seed, pages):
    def get_domain(url):
        # Host portion: between "://" and next "/", or up to first "/" if no "://"
        if '://' in url:
            rest = url.split('://', 1)[1]
        else:
            rest = url
        return rest.split('/', 1)[0]
    
    seed_domain = get_domain(seed)
    seen = {seed}
    visited = []
    queue = deque([seed])
    
    while queue:
        url = queue.popleft()
        visited.append(url)
        for link in pages.get(url, []):
            if link in seen:
                continue
            if get_domain(link) != seed_domain:
                continue
            seen.add(link)
            queue.append(link)
    
    return visited
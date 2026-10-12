import hashlib
import re


def canonical(src, name):
    pattern = r'\b' + re.escape(name) + r'\b'
    return re.sub(pattern, 'K', src)


class KernelCache:
    def __init__(self):
        self.compiles = 0
        self.hits = 0
        self.sources = {}

    def compile(self, src, name):
        canon = canonical(src, name)
        key = hashlib.sha1(canon.encode()).hexdigest()
        if key in self.sources:
            self.hits += 1
        else:
            self.compiles += 1
            self.sources[key] = canon
        return key


def dedupe_program(kernels):
    cache = KernelCache()
    calls = []
    for kernel in kernels:
        key = cache.compile(kernel['src'], kernel['name'])
        calls.append((key, kernel['bufs']))
    unique_sources = len(cache.sources)
    return unique_sources, calls


def compile_savings(kernels, compile_ms):
    unique_sources, _ = dedupe_program(kernels)
    return (len(kernels) - unique_sources) * compile_ms
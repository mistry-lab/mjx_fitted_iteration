import math
import shutil
import sys
import time
from typing import Optional

class tqdm:
    def __init__(self, iterable=None, desc: str = '', disable: bool = False, unit: str = 'it', unit_scale=False,
                 total: Optional[int] = None, rate: int = 100):
        self.iterable = iterable
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.rate = rate
        self.st = time.perf_counter()
        self.i = -1
        self.n = 0
        self.skip = 1
        self.t = getattr(iterable, "__len__",  lambda: 0)() if total is None else total
        self.postfix = {}
        self.set_description(desc)
        self.update(0)

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.update(close=True)

    def set_description(self, desc: str):
        self.desc = f"{desc}: " if desc else ""

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        if ordered_dict:
            self.postfix.update(ordered_dict)
        self.postfix.update(kwargs)
        if refresh:
            self.update(0)

    def update(self, n: int = 0, close: bool = False):
        self.n, self.i = self.n + n, self.i + 1
        if self.disable or (not close and self.i % self.skip != 0):
            return
        prog, elapsed, ncols = self.n / self.t if self.t else 0, time.perf_counter() - self.st, shutil.get_terminal_size().columns
        if self.i / elapsed > self.rate and self.i:
            self.skip = max(int(self.i / elapsed) // self.rate, 1)

        def HMS(t):
            return ':'.join(
                f'{x:02d}' if i else str(x) for i, x in enumerate([int(t) // 3600, int(t) % 3600 // 60, int(t) % 60]) if
                i or x)

        def SI(x):
            return (f"{x / 1000 ** int(g := math.log(x, 1000)):.{int(3 - 3 * math.fmod(g, 1))}f}"[:4].rstrip('.') +
                    ' kMGTPEZY'[int(g)].strip()) if x else '0.00'

        prog_text = f'{SI(self.n)}{f"/{SI(self.t)}" if self.t else self.unit}' if self.unit_scale else f'{self.n}{f"/{self.t}" if self.t else self.unit}'
        elapsed_text = HMS(elapsed) + (f'<{HMS(elapsed / prog - elapsed) if self.n else "?"}' if self.t else '')
        it_text = (SI(self.n / elapsed) if self.unit_scale else f"{self.n / elapsed:5.2f}") if self.n else "?"
        suf = f'{prog_text} [{elapsed_text}, {it_text}{self.unit}/s]'

        # Include postfix information
        postfix_str = ''
        if self.postfix:
            postfix_str = ' | ' + ', '.join(f'{k}={v}' for k, v in self.postfix.items())

        sz = max(ncols - len(self.desc) - 3 - 2 - 2 - len(suf + postfix_str), 1)
        num = sz * prog
        bar_chars = "█" * int(num)
        partial_char = " ▏▎▍▌▋▊▉"[int(8 * num) % 8].strip()
        bar = '\r' + self.desc + (
            f'{100 * prog:3.0f}%|{(bar_chars + partial_char).ljust(sz, " ")}| ' if self.t else '') + suf + postfix_str
        print(bar[:ncols + 1], flush=True, end='\r' * close, file=sys.stderr)


class trange(tqdm):
    def __init__(self, n: int, **kwargs):
        super().__init__(iterable=range(n), total=n, **kwargs)

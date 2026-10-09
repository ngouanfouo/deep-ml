import numpy as np


class View:
    def __init__(self, shape, strides, offset=0, mask=None):
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.offset = offset
        if mask is None:
            self.mask = None
        else:
            self.mask = tuple(tuple(m) for m in mask)

    @staticmethod
    def contiguous(shape):
        shape = tuple(shape)
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return View(shape, tuple(strides), 0, None)

    def offset_of(self, idx):
        return self.offset + sum(i * s for i, s in zip(idx, self.strides))

    def valid(self, idx):
        if self.mask is None:
            return True
        for i, (lo, hi) in zip(idx, self.mask):
            if not (lo <= i < hi):
                return False
        return True

    def is_contiguous(self):
        if self.mask is not None or self.offset != 0:
            return False
        expected = [1] * len(self.shape)
        for i in range(len(self.shape) - 2, -1, -1):
            expected[i] = expected[i + 1] * self.shape[i + 1]
        return self.strides == tuple(expected)

    def permute(self, perm):
        perm = tuple(perm)
        mask = None if self.mask is None else tuple(self.mask[p] for p in perm)
        return View(
            tuple(self.shape[p] for p in perm),
            tuple(self.strides[p] for p in perm),
            self.offset,
            mask,
        )

    def expand(self, shape):
        shape = tuple(shape)
        new_strides = []
        new_mask = [] if self.mask is not None else None
        for i, (old_s, new_s) in enumerate(zip(self.shape, shape)):
            if old_s == 1 and new_s != 1:
                new_strides.append(0)
                if new_mask is not None:
                    lo, hi = self.mask[i]
                    new_mask.append((0, new_s) if (lo, hi) == (0, 1) else (lo, hi))
            else:
                new_strides.append(self.strides[i])
                if new_mask is not None:
                    new_mask.append(self.mask[i])
        return View(
            shape,
            tuple(new_strides),
            self.offset,
            None if new_mask is None else tuple(new_mask),
        )

    def flip(self, axes):
        axes = set(axes)
        new_strides = list(self.strides)
        new_mask = list(self.mask) if self.mask is not None else None
        new_offset = self.offset
        for a in axes:
            s = self.shape[a]
            new_offset += (s - 1) * self.strides[a]
            new_strides[a] = -self.strides[a]
            if new_mask is not None:
                lo, hi = self.mask[a]
                new_mask[a] = (s - hi, s - lo)
        return View(
            self.shape,
            tuple(new_strides),
            new_offset,
            None if new_mask is None else tuple(new_mask),
        )

    def shrink(self, ranges):
        new_shape = []
        new_mask = [] if self.mask is not None else None
        new_offset = self.offset
        for a, (start, end) in enumerate(ranges):
            new_offset += start * self.strides[a]
            new_shape.append(end - start)
            if new_mask is not None:
                lo, hi = self.mask[a]
                new_mask.append((max(0, lo - start), min(end - start, hi - start)))
        return View(
            tuple(new_shape),
            self.strides,
            new_offset,
            None if new_mask is None else tuple(new_mask),
        )

    def pad(self, pads):
        new_shape = []
        new_mask = []
        new_offset = self.offset
        for a, (lo, hi) in enumerate(pads):
            old_size = self.shape[a]
            new_offset -= lo * self.strides[a]
            new_shape.append(old_size + lo + hi)
            valid_lo, valid_hi = lo, lo + old_size
            if self.mask is not None:
                mlo, mhi = self.mask[a]
                valid_lo = max(valid_lo, mlo + lo)
                valid_hi = min(valid_hi, mhi + lo)
            new_mask.append((valid_lo, valid_hi))
        return View(tuple(new_shape), self.strides, new_offset, tuple(new_mask))


def gather(view, buf):
    out = np.zeros(view.shape, dtype=np.float32)
    for idx in np.ndindex(view.shape):
        if view.valid(idx):
            out[idx] = buf[view.offset_of(idx)]
    return out
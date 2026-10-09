import ctypes
import os
import shutil
import subprocess
import tempfile
import numpy as np


def _row_major_strides(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


def _flat_expr(terms):
    # terms: list of (var_name, stride) -> "r0 * 4 + r1 * 1"
    parts = []
    for name, stride in terms:
        parts.append(f"{name} * {stride}")
    return " + ".join(parts) if parts else "0"


def render_reduce(shape, axes, op="sum", name="reduce"):
    shape = tuple(shape)
    axes = tuple(axes)
    axes_set = set(axes)

    in_strides = _row_major_strides(shape)

    kept_axes = [a for a in range(len(shape)) if a not in axes_set]
    kept_shape = tuple(shape[a] for a in kept_axes)
    out_strides = _row_major_strides(kept_shape)

    lines = []
    lines.append(
        f"void {name}(float* restrict out, const float* restrict in) {{"
    )

    indent = 1

    def emit(s):
        lines.append("  " * indent + s)

    # Output loops (over kept axes, in order)
    for a in kept_axes:
        emit(f"for (int r{a} = 0; r{a} < {shape[a]}; r{a}++) {{")
        indent += 1

    # Accumulator
    init = "0.0f" if op == "sum" else "-INFINITY"
    emit(f"float acc = {init};")

    # Reduce loops (over reduced axes, in order)
    for a in axes:
        emit(f"for (int r{a} = 0; r{a} < {shape[a]}; r{a}++) {{")
        indent += 1

    # Update
    flat_terms = [(f"r{i}", in_strides[i]) for i in range(len(shape))]
    flat = _flat_expr(flat_terms)
    if op == "sum":
        emit(f"acc = acc + in[{flat}];")
    else:
        emit(f"fmaxf(acc, in[{flat}]);")  # placeholder; replaced below

    # Close reduce loops
    for _ in axes:
        indent -= 1
        emit("}")

    # Store result
    out_terms = [(f"r{a}", out_strides[j]) for j, a in enumerate(kept_axes)]
    flat_out = _flat_expr(out_terms)
    emit(f"out[{flat_out}] = acc;")

    # Close output loops
    for _ in kept_axes:
        indent -= 1
        emit("}")

    lines.append("}")
    src = "\n".join(lines) + "\n"

    # Fix the max update line to the correct assignment
    if op == "max":
        src = src.replace("fmaxf(acc, in[", "acc = fmaxf(acc, in[")

    return src


def run_reduce(shape, axes, x, op="sum"):
    src = render_reduce(shape, axes, op=op, name="reduce")
    full = "#include <math.h>\n" + src

    shape = tuple(shape)
    axes = tuple(axes)
    kept_shape = tuple(s for a, s in enumerate(shape) if a not in set(axes))

    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros(kept_shape, dtype=np.float32)

    tmpdir = tempfile.mkdtemp()
    try:
        cpath = os.path.join(tmpdir, "k.c")
        sopath = os.path.join(tmpdir, "k.so")
        with open(cpath, "w") as f:
            f.write(full)
        subprocess.run(
            ["cc", "-O2", "-shared", "-fPIC", "-w", "-lm", cpath, "-o", sopath],
            check=True,
        )
        lib = ctypes.CDLL(sopath)
        fn = lib.reduce
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        fn(
            out.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p),
        )
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
import ctypes
import os
import re
import shutil
import subprocess
import tempfile
import numpy as np


def render(program, name="kernel"):
    buf_order = []
    seen_bufs = set()

    def add_buf(b):
        if b not in seen_bufs:
            seen_bufs.add(b)
            buf_order.append(b)

    def collect_bufs(expr):
        if isinstance(expr, tuple):
            if expr[0] == "load":
                add_buf(expr[1])
                return
            for sub in expr[1:]:
                collect_bufs(sub)

    for store in program:
        add_buf(store[1])
        collect_bufs(store[3])

    lines = []
    var_map = {}
    counter = [0]

    def lit(v):
        return f"{float(v)}f"

    def emit(expr):
        if isinstance(expr, tuple):
            if expr in var_map:
                return var_map[expr]
            op = expr[0]
            if op == "load":
                typ, rhs = "float", f"{expr[1]}[{expr[2]}]"
            elif op == "lt":
                a, b = emit(expr[1]), emit(expr[2])
                typ, rhs = "int", f"({a} < {b})"
            elif op == "where":
                c, a, b = emit(expr[1]), emit(expr[2]), emit(expr[3])
                typ, rhs = "float", f"({c} ? {a} : {b})"
            elif op == "add":
                a, b = emit(expr[1]), emit(expr[2])
                typ, rhs = "float", f"({a} + {b})"
            elif op == "mul":
                a, b = emit(expr[1]), emit(expr[2])
                typ, rhs = "float", f"({a} * {b})"
            elif op == "max":
                a, b = emit(expr[1]), emit(expr[2])
                typ, rhs = "float", f"fmaxf({a}, {b})"
            elif op == "exp2":
                a = emit(expr[1])
                typ, rhs = "float", f"exp2f({a})"
            elif op == "recip":
                a = emit(expr[1])
                typ, rhs = "float", f"(1.0f/{a})"
            else:
                raise ValueError(f"unknown op: {op}")

            v = f"v{counter[0]}"
            counter[0] += 1
            var_map[expr] = v
            lines.append(f"  {typ} {v} = {rhs};")
            return v
        else:
            return lit(expr)

    for store in program:
        v = emit(store[3])
        lines.append(f"  {store[1]}[{store[2]}] = {v};")

    params = ", ".join(f"float* restrict {b}" for b in buf_order)
    header = f"void {name}({params}) {{"
    return "\n".join([header] + lines + ["}"]) + "\n"


def compile_and_run(src, name, bufs):
    full_src = "#include <math.h>\n" + src

    tmpdir = tempfile.mkdtemp()
    try:
        cpath = os.path.join(tmpdir, "k.c")
        sopath = os.path.join(tmpdir, "k.so")
        with open(cpath, "w") as f:
            f.write(full_src)

        proc = subprocess.run(
            ["cc", "-O2", "-shared", "-fPIC", "-w", "-lm", cpath, "-o", sopath],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)

        lib = ctypes.CDLL(sopath)

        m = re.search(r"void\s+\w+\s*\(([^)]*)\)", src)
        params_str = m.group(1)
        buf_names = [
            p.strip().split()[-1]
            for p in params_str.split(",")
            if p.strip()
        ]

        func = getattr(lib, name)
        func.restype = None

        ptrs = [
            np.ascontiguousarray(bufs[b], dtype=np.float32).ctypes.data_as(
                ctypes.c_void_p
            )
            for b in buf_names
        ]
        func(*ptrs)
        return bufs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
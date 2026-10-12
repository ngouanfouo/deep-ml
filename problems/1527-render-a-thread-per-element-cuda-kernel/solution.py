def render_cuda(desc):
    name = desc['name']
    out_shape = desc['out_shape']
    reduce = desc.get('reduce', [])
    params = desc['params']
    body = desc['body']
    
    L = len(out_shape)
    n0 = out_shape[-1] if L >= 1 else 1
    n1 = out_shape[-2] if L >= 2 else 1
    n2 = out_shape[-3] if L >= 3 else 1
    
    param_decls = []
    for i, p in enumerate(params):
        if i == 0:
            param_decls.append(f"float* {p}")
        else:
            param_decls.append(f"const float* {p}")
    param_str = ", ".join(param_decls)
    
    lines = []
    lines.append(f"__global__ void {name}({param_str}) {{")
    
    if L >= 1:
        lines.append("  int gidx0 = blockIdx.x * blockDim.x + threadIdx.x;")
    if L >= 2:
        lines.append("  int gidx1 = blockIdx.y * blockDim.y + threadIdx.y;")
    if L >= 3:
        lines.append("  int gidx2 = blockIdx.z * blockDim.z + threadIdx.z;")
    
    guard_terms = []
    if L >= 1:
        guard_terms.append(f"gidx0 < {n0}")
    if L >= 2:
        guard_terms.append(f"gidx1 < {n1}")
    if L >= 3:
        guard_terms.append(f"gidx2 < {n2}")
    guard_cond = " && ".join(guard_terms)
    lines.append(f"  if ({guard_cond}) {{")
    
    if L == 1:
        flat = "gidx0"
    elif L == 2:
        flat = f"gidx1 * {n0} + gidx0"
    elif L >= 3:
        flat = f"gidx2 * {n1 * n0} + gidx1 * {n0} + gidx0"
    else:
        flat = "gidx0"
    
    if reduce:
        lines.append("    float acc = 0.0f;")
        indent = 4
        for var, ext in reduce:
            lines.append(" " * indent + f"for (int {var} = 0; {var} < {ext}; {var}++) {{")
            indent += 2
        lines.append(" " * indent + f"acc += {body};")
        for _ in reduce:
            indent -= 2
            lines.append(" " * indent + "}")
        lines.append(" " * indent + f"{params[0]}[{flat}] = acc;")
    else:
        lines.append("    " + f"{params[0]}[{flat}] = {body};")
    
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def launch_config(desc, block=(32, 4, 1)):
    out_shape = desc['out_shape']
    L = len(out_shape)
    ex = out_shape[-1] if L >= 1 else 1
    ey = out_shape[-2] if L >= 2 else 1
    ez = out_shape[-3] if L >= 3 else 1
    
    bx, by, bz = block
    bx = min(bx, ex)
    by = min(by, ey)
    bz = min(bz, ez)
    
    gx = -(-ex // bx)
    gy = -(-ey // by)
    gz = -(-ez // bz)
    
    return (gx, gy, gz), (bx, by, bz)


def cuda_launcher(desc, block=(32, 4, 1)):
    name = desc['name']
    params = desc['params']
    grid, blk = launch_config(desc, block)
    
    param_decls = []
    for i, p in enumerate(params):
        if i == 0:
            param_decls.append(f"float* {p}")
        else:
            param_decls.append(f"const float* {p}")
    param_str = ", ".join(param_decls)
    param_names = ", ".join(params)
    
    lines = []
    lines.append(f'extern "C" void launch_{name}({param_str}) {{')
    lines.append(f"  dim3 grid({grid[0]}, {grid[1]}, {grid[2]}), block({blk[0]}, {blk[1]}, {blk[2]});")
    lines.append(f"  {name}<<<grid, block>>>({param_names});")
    lines.append("  cudaDeviceSynchronize();")
    lines.append("}")
    return "\n".join(lines)
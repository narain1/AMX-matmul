# Converting Intel AMX MatMul to Tinygrad UOp Kernels

## Introduction

This guide explains how to convert Intel AMX (Advanced Matrix Extensions) matrix multiplication operations to tinygrad's Universal Operation (UOp) kernel representation.

## Understanding UOp Singleton Pattern

UOp follows a singleton pattern where identical operations with the same parameters return the same instance. This enables efficient AST comparison and transformation.

```python
from tinygrad.ops import UOp, Ops
from tinygrad.dtype import dtypes

# Same parameters = same instance
const1 = UOp(Ops.CONST, dtypes.float, arg=0.5)
const2 = UOp(Ops.CONST, dtypes.float, arg=0.5)
print(const1 == const2)  # True (same object reference)
```

## AMX Operations Mapping

### AMX Tile Representation

AMX tiles are 16x32 BF16 elements (1KB) that can be represented as UOp buffers:

```python
from tinygrad.ops import UOp, Ops
from tinygrad.dtype import dtypes

# Define AMX tile as a buffer (16 rows x 32 columns of BF16)
def create_amx_tile(buffer_id: int, rows: int = 16, cols: int = 32, dtype=dtypes.bfloat16):
    """Create a UOp representing an AMX tile buffer"""
    shape = (rows, cols)
    return UOp(Ops.DEFINE_GLOBAL, dtype, arg=(buffer_id, shape))

# Create tiles A, B, C
tile_a = create_amx_tile(1)  # Input tile A
tile_b = create_amx_tile(2)  # Input tile B  
tile_c = create_amx_tile(3, rows=16, cols=16, dtype=dtypes.float32)  # Output tile C (FP32)
```

### AMX Load Operations (tileloadd)

The `tileloadd` instruction loads data from memory into a tile register:

```python
def amx_tile_load(buffer: UOp, offset: int = 0, stride: int = 64):
    """
    Represent AMX tileloadd operation
    Assembly: tileloadd tmm0, [rcx + r10]
    """
    # Create offset UOp
    offset_uop = UOp(Ops.CONST, dtypes.int32, arg=offset)
    
    # Create load operation
    load_op = UOp(Ops.LOAD, buffer.dtype, src=(buffer, offset_uop))
    
    return load_op

# Example: Load tile from buffer
tile_a_data = amx_tile_load(tile_a, offset=0)
```

### AMX Matrix Multiply (tdpbf16ps)

The `tdpbf16ps` instruction performs: C[16x16 FP32] += A[16x32 BF16] * B[16x32 BF16]

```python
def amx_tdpbf16ps(c_tile: UOp, a_tile: UOp, b_tile: UOp):
    """
    Represent AMX tdpbf16ps operation
    Assembly: tdpbf16ps tmm0, tmm1, tmm2
    Operation: C[16x16] += A[16x32] * B[16x32]
    
    The operation performs:
    FOR m := 0 TO 15
        FOR k := 0 TO 15
            FOR n := 0 TO 15
                c[m][n] += FP32(a[m][2*k+0]) * FP32(b[k][2*n+0])
                c[m][n] += FP32(a[m][2*k+1]) * FP32(b[k][2*n+1])
    """
    # Matrix multiply operation
    matmul = UOp(Ops.WMMA, dtypes.float32, src=(a_tile, b_tile))
    
    # Accumulate into C (C += A * B)
    result = UOp(Ops.ADD, dtypes.float32, src=(c_tile, matmul))
    
    return result
```

### AMX Store Operations (tilestored)

The `tilestored` instruction stores a tile register back to memory:

```python
def amx_tile_store(buffer: UOp, data: UOp, offset: int = 0):
    """
    Represent AMX tilestored operation
    Assembly: tilestored [rcx + r10], tmm0
    """
    offset_uop = UOp(Ops.CONST, dtypes.int32, arg=offset)
    
    # Create store operation
    store_op = UOp(Ops.STORE, dtypes.void, src=(buffer, offset_uop, data))
    
    return store_op
```

## Complete Example: 16x16x32 MatMul

Converting the simple AMX assembly to UOp representation:

```asm
; Assembly version (tdpbf16ps_N16_M16_K32_asm)
tdpbf16ps_N16_M16_K32_asm:
    mov         r10d, 64
    ldtilecfg   [config]
    tileloadd   tmm0, [rcx + r10]  ; load C[0][0]
    tileloadd   tmm1, [rdx + r10]  ; load A[0][0]
    tileloadd   tmm2, [r8 + r10]   ; load B[0][0]
    tdpbf16ps   tmm0, tmm1, tmm2   ; C[0][0] += A[0][0] * B[0][0]
    tilestored  [rcx + r10], tmm0  ; store C[0][0]
    tilerelease
    ret
```

UOp representation:

```python
from tinygrad.ops import UOp, Ops
from tinygrad.dtype import dtypes

def amx_matmul_16x16x32_uop():
    """
    UOp representation of tdpbf16ps_N16_M16_K32_asm
    Performs: C[16x16 FP32] += A[16x32 BF16] * B[16x32 BF16]
    """
    # Define buffers for C, A, B matrices
    buf_c = UOp(Ops.DEFINE_GLOBAL, dtypes.float32, arg=(0, (16, 16)))
    buf_a = UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16, arg=(1, (16, 32)))
    buf_b = UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16, arg=(2, (16, 32)))
    
    # Stride constant
    stride = UOp(Ops.CONST, dtypes.int32, arg=64)
    
    # Load operations
    c_loaded = UOp(Ops.LOAD, dtypes.float32, src=(buf_c, stride))
    a_loaded = UOp(Ops.LOAD, dtypes.bfloat16, src=(buf_a, stride))
    b_loaded = UOp(Ops.LOAD, dtypes.bfloat16, src=(buf_b, stride))
    
    # Matrix multiplication: result = A * B
    matmul_result = UOp(Ops.WMMA, dtypes.float32, src=(a_loaded, b_loaded))
    
    # Accumulate: C = C + (A * B)
    c_updated = UOp(Ops.ADD, dtypes.float32, src=(c_loaded, matmul_result))
    
    # Store result back
    store_op = UOp(Ops.STORE, dtypes.void, src=(buf_c, stride, c_updated))
    
    return store_op

# Create the UOp kernel
kernel = amx_matmul_16x16x32_uop()
print(kernel)
```

## Complex Example: 32x32x64 MatMul with Tile Reuse

For larger matrices, AMX uses multiple tiles. Here's the pattern from `tdpbf16ps_N32_M32_K64_asm`:

```python
def amx_matmul_32x32x64_uop():
    """
    UOp representation of tdpbf16ps_N32_M32_K64_asm
    Performs 8 tile operations with optimal register reuse:
    
    C[0][1] += A[0][1] * B[0][0]
    C[0][0] += A[0][0] * B[0][0]
    C[1][0] += A[0][0] * B[0][1]
    C[1][1] += A[0][1] * B[0][1]
    C[0][0] += A[1][0] * B[1][0]
    C[0][1] += A[1][1] * B[1][0]
    C[1][0] += A[1][0] * B[1][1]
    C[1][1] += A[1][1] * B[1][1]
    """
    stride = UOp(Ops.CONST, dtypes.int32, arg=64)
    tile_offset = UOp(Ops.CONST, dtypes.int32, arg=1024)
    
    # Define tile buffers
    # C has 4 tiles (2x2), A has 4 tiles (2x2), B has 4 tiles (2x2)
    c_tiles = [[UOp(Ops.DEFINE_GLOBAL, dtypes.float32, arg=(f"C_{i}_{j}", (16, 16))) 
                for j in range(2)] for i in range(2)]
    a_tiles = [[UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16, arg=(f"A_{i}_{j}", (16, 32))) 
                for j in range(2)] for i in range(2)]
    b_tiles = [[UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16, arg=(f"B_{i}_{j}", (16, 32))) 
                for j in range(2)] for i in range(2)]
    
    # First K iteration (k=0)
    # Load A[0][1], B[0][0], C[0][1]
    a_01_loaded = UOp(Ops.LOAD, dtypes.bfloat16, src=(a_tiles[0][1], stride))
    b_00_loaded = UOp(Ops.LOAD, dtypes.bfloat16, src=(b_tiles[0][0], stride))
    c_01_loaded = UOp(Ops.LOAD, dtypes.float32, src=(c_tiles[0][1], stride))
    
    # C[0][1] += A[0][1] * B[0][0]
    matmul_01 = UOp(Ops.WMMA, dtypes.float32, src=(a_01_loaded, b_00_loaded))
    c_01_updated = UOp(Ops.ADD, dtypes.float32, src=(c_01_loaded, matmul_01))
    
    # Load A[0][0], C[0][0]
    a_00_loaded = UOp(Ops.LOAD, dtypes.bfloat16, src=(a_tiles[0][0], stride))
    c_00_loaded = UOp(Ops.LOAD, dtypes.float32, src=(c_tiles[0][0], stride))
    
    # C[0][0] += A[0][0] * B[0][0] (reuse b_00_loaded)
    matmul_00 = UOp(Ops.WMMA, dtypes.float32, src=(a_00_loaded, b_00_loaded))
    c_00_updated = UOp(Ops.ADD, dtypes.float32, src=(c_00_loaded, matmul_00))
    
    # Load B[0][1], C[1][0]
    b_01_loaded = UOp(Ops.LOAD, dtypes.bfloat16, src=(b_tiles[0][1], stride))
    c_10_loaded = UOp(Ops.LOAD, dtypes.float32, src=(c_tiles[1][0], stride))
    
    # C[1][0] += A[0][0] * B[0][1] (reuse a_00_loaded)
    matmul_10 = UOp(Ops.WMMA, dtypes.float32, src=(a_00_loaded, b_01_loaded))
    c_10_updated = UOp(Ops.ADD, dtypes.float32, src=(c_10_loaded, matmul_10))
    
    # Load C[1][1]
    c_11_loaded = UOp(Ops.LOAD, dtypes.float32, src=(c_tiles[1][1], stride))
    
    # C[1][1] += A[0][1] * B[0][1] (reuse a_01_loaded, b_01_loaded)
    matmul_11 = UOp(Ops.WMMA, dtypes.float32, src=(a_01_loaded, b_01_loaded))
    c_11_updated_k0 = UOp(Ops.ADD, dtypes.float32, src=(c_11_loaded, matmul_11))
    
    # Second K iteration (k=1) - similar pattern with A[1][*] and B[1][*]
    # ... (abbreviated for clarity)
    
    # Store all results
    stores = []
    for i in range(2):
        for j in range(2):
            # In practice, would reference the updated C tiles
            store = UOp(Ops.STORE, dtypes.void, src=(c_tiles[i][j], stride, c_00_updated))
            stores.append(store)
    
    return stores

# The UOp tree represents the entire computation
kernel = amx_matmul_32x32x64_uop()
```

## Dynamic MatMul Algorithms

For variable-sized matrices, the AMX implementations use loops. Here's how to represent the `amx2` algorithm:

```python
def amx2_dynamic_matmul_uop(M: int, N: int, K: int):
    """
    UOp representation of the AMX2 algorithm from README.md
    Maximizes tile reuse with 4 C tiles and 2 A, 2 B tiles
    
    for i in range(0, M, 2):
        for j in range(0, N, 2):
            # Load 4 C tiles
            # For each k in K:
            #   Load 2 A tiles, 2 B tiles
            #   Perform 4 multiplications
            # Store 4 C tiles
    """
    stride = UOp(Ops.CONST, dtypes.int32, arg=64)
    
    # Loop indices as UOps
    i_range = UOp(Ops.RANGE, dtypes.int32, arg=(0, M, 2))
    j_range = UOp(Ops.RANGE, dtypes.int32, arg=(0, N, 2))
    k_range = UOp(Ops.RANGE, dtypes.int32, arg=(0, K, 1))
    
    # Define buffers
    buf_c = UOp(Ops.DEFINE_GLOBAL, dtypes.float32, arg=(0, (M, N)))
    buf_a = UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16, arg=(1, (M, K)))
    buf_b = UOp(Ops.DEFINE_GLOBAL, dtypes.bfloat16, arg=(2, (K, N)))
    
    # Create loop structure
    # i loop
    i_loop = UOp(Ops.LOOP, dtypes.void, src=(i_range,))
    i_idx = UOp(Ops.INDEX, dtypes.int32, src=(i_loop,), arg=0)
    
    # j loop (nested in i)
    j_loop = UOp(Ops.LOOP, dtypes.void, src=(j_range, i_loop))
    j_idx = UOp(Ops.INDEX, dtypes.int32, src=(j_loop,), arg=0)
    
    # Load 4 C tiles: C[i:i+2, j:j+2]
    c_00_idx = UOp(Ops.ADD, dtypes.int32, src=(i_idx, j_idx))
    c_00 = UOp(Ops.LOAD, dtypes.float32, src=(buf_c, c_00_idx))
    
    # k loop (innermost)
    k_loop = UOp(Ops.LOOP, dtypes.void, src=(k_range, j_loop))
    k_idx = UOp(Ops.INDEX, dtypes.int32, src=(k_loop,), arg=0)
    
    # Load A[i:i+2, k] and B[k, j:j+2]
    a_idx = UOp(Ops.ADD, dtypes.int32, src=(i_idx, k_idx))
    a = UOp(Ops.LOAD, dtypes.bfloat16, src=(buf_a, a_idx))
    
    b_idx = UOp(Ops.ADD, dtypes.int32, src=(k_idx, j_idx))
    b = UOp(Ops.LOAD, dtypes.bfloat16, src=(buf_b, b_idx))
    
    # Perform multiplication: C += A * B
    matmul = UOp(Ops.WMMA, dtypes.float32, src=(a, b))
    c_updated = UOp(Ops.ADD, dtypes.float32, src=(c_00, matmul))
    
    # Store result
    store = UOp(Ops.STORE, dtypes.void, src=(buf_c, c_00_idx, c_updated))
    
    return store
```

## Key Concepts Summary

### AMX Operation → UOp Mapping

| AMX Operation | UOp Representation | Notes |
|---------------|-------------------|-------|
| `tileloadd` | `UOp(Ops.LOAD, ...)` | Load tile from memory |
| `tilestored` | `UOp(Ops.STORE, ...)` | Store tile to memory |
| `tdpbf16ps` | `UOp(Ops.WMMA, ...) + UOp(Ops.ADD, ...)` | Matrix multiply-accumulate |
| `ldtilecfg` | Configuration in buffer definitions | Tile config encoded in buffer shape |
| `tilerelease` | Implicit at end of kernel | Resource cleanup |

### Benefits of UOp Representation

1. **Algebraic Simplification**: UOp's singleton pattern enables easy pattern matching and optimization
2. **Platform Independence**: Same UOp tree can target different backends (AMX, GPU, CPU)
3. **Composability**: Complex operations built from simple primitives
4. **Optimization**: Compiler can optimize the UOp tree before code generation

### Example: Comparing Two MatMul Patterns

```python
# Check if two matrix multiply patterns are equivalent
def remove_buffer_loads(uop: UOp):
    """Remove LOAD operations to compare computational patterns"""
    src = [remove_buffer_loads(u) for u in uop.src]
    src = tuple([u for u in src if u is not None])
    if uop.op == Ops.LOAD:
        return None
    return uop.replace(src=src)

# Create two different matmul implementations
impl1 = amx_matmul_16x16x32_uop()
impl2 = amx_matmul_16x16x32_uop()  # Same implementation

# Compare without memory operations
pattern1 = remove_buffer_loads(impl1)
pattern2 = remove_buffer_loads(impl2)

print(pattern1 == pattern2)  # True - same computation pattern
```

## Practical Tips

1. **Start Simple**: Begin with single-tile operations (16x16x32) before tackling multi-tile
2. **Preserve Memory Layout**: AMX has specific memory layout requirements (e.g., BF16 packing in B matrix)
3. **Optimize for Reuse**: Like the assembly code, minimize loads by reusing tiles
4. **Use Type Information**: BF16 vs FP32 is critical for AMX operations
5. **Test Incrementally**: Build and verify each operation before composing larger kernels

## Further Reading

- [Tinygrad Documentation](https://github.com/tinygrad/tinygrad)
- [Intel AMX Programming Guide](https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-matrix-extensions-intel-amx-instructions.html)
- [AMX MatMul Repository](https://github.com/narain1/AMX-matmul)

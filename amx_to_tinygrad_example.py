#!/usr/bin/env python3
"""
Example: Converting Intel AMX MatMul to Tinygrad UOp Kernels

This example demonstrates how to represent AMX matrix multiplication
operations using tinygrad's Universal Operation (UOp) representation.

Based on the AMX implementations from this repository, particularly:
- tdpbf16ps_N16_M16_K32.asm (simple single-tile matmul)
- tdpbf16ps_N32_M32_K64.asm (multi-tile with register reuse)
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass


# Mock tinygrad imports for demonstration
# In practice, you would use: from tinygrad.ops import UOp, Ops
# from tinygrad.dtype import dtypes

class Ops:
    """Mock Ops enum - in tinygrad these are the operation types"""
    CONST = "CONST"
    DEFINE_GLOBAL = "DEFINE_GLOBAL"
    LOAD = "LOAD"
    STORE = "STORE"
    WMMA = "WMMA"  # Warp Matrix Multiply Accumulate
    ADD = "ADD"
    LOOP = "LOOP"
    INDEX = "INDEX"
    RANGE = "RANGE"
    

class DTypes:
    """Mock DTypes - in tinygrad these are the data types"""
    bfloat16 = "bfloat16"
    float32 = "float32"
    int32 = "int32"
    void = "void"


dtypes = DTypes()


class UOp:
    """
    Mock UOp class demonstrating the singleton pattern.
    
    In tinygrad, UOp uses a metaclass to implement the singleton pattern:
    - Same (op, dtype, src, arg) → same instance
    - Enables efficient tree comparison with ==
    """
    
    # Simplified singleton cache (real tinygrad uses weakref)
    _cache = {}
    _next_id = 0
    
    def __init__(self, op, dtype, src=(), arg=None):
        # Only initialize once due to singleton
        if hasattr(self, '_initialized'):
            return
        self.op = op
        self.dtype = dtype
        self.src = tuple(src) if src else ()
        self.arg = arg
        self._initialized = True
        # Assign unique ID for hashing
        self._id = UOp._next_id
        UOp._next_id += 1
    
    def __new__(cls, op, dtype, src=(), arg=None):
        # Convert src to tuple of IDs for hashable key
        src_tuple = tuple(src) if src else ()
        src_ids = tuple(id(s) for s in src_tuple)
        
        # Make arg hashable
        if isinstance(arg, (list, dict)):
            arg_key = str(arg)
        else:
            arg_key = arg
            
        key = (op, dtype, src_ids, arg_key)
        
        if key not in cls._cache:
            instance = super().__new__(cls)
            cls._cache[key] = instance
        return cls._cache[key]
    
    def replace(self, **kwargs):
        """Create a new UOp with some attributes replaced"""
        return UOp(
            op=kwargs.get('op', self.op),
            dtype=kwargs.get('dtype', self.dtype),
            src=kwargs.get('src', self.src),
            arg=kwargs.get('arg', self.arg)
        )
    
    def __hash__(self):
        """Enable UOp to be used in sets and as dict keys"""
        return self._id
    
    def __eq__(self, other):
        """Singleton instances are only equal to themselves"""
        return self is other
    
    def __repr__(self):
        arg_str = f", arg={self.arg}" if self.arg is not None else ""
        if self.src:
            src_str = ",\n    ".join(repr(s) for s in self.src)
            return f"UOp({self.op}, {self.dtype}{arg_str}, src=(\n    {src_str}))"
        return f"UOp({self.op}, {self.dtype}{arg_str})"


# ==============================================================================
# AMX Tile Primitives
# ==============================================================================

def create_amx_tile(buffer_id: int, rows: int = 16, cols: int = 32, 
                    dtype=dtypes.bfloat16) -> UOp:
    """
    Create a UOp representing an AMX tile buffer.
    
    AMX tiles are:
    - 16 rows x 32 columns for BF16 (1KB)
    - 16 rows x 16 columns for FP32 (1KB)
    
    Args:
        buffer_id: Unique identifier for this buffer
        rows: Number of rows (typically 16)
        cols: Number of columns (32 for BF16, 16 for FP32)
        dtype: Data type (bfloat16 or float32)
    
    Returns:
        UOp representing the buffer
    """
    shape = (rows, cols)
    return UOp(Ops.DEFINE_GLOBAL, dtype, arg=(buffer_id, shape))


def amx_tile_load(buffer: UOp, offset: int = 0) -> UOp:
    """
    AMX tileloadd operation - load tile from memory.
    
    Assembly equivalent: tileloadd tmm0, [addr + offset]
    
    Args:
        buffer: Buffer to load from
        offset: Byte offset
    
    Returns:
        UOp representing the loaded data
    """
    offset_uop = UOp(Ops.CONST, dtypes.int32, arg=offset)
    return UOp(Ops.LOAD, buffer.dtype, src=(buffer, offset_uop))


def amx_tile_store(buffer: UOp, data: UOp, offset: int = 0) -> UOp:
    """
    AMX tilestored operation - store tile to memory.
    
    Assembly equivalent: tilestored [addr + offset], tmm0
    
    Args:
        buffer: Buffer to store to
        data: Data to store
        offset: Byte offset
    
    Returns:
        UOp representing the store operation
    """
    offset_uop = UOp(Ops.CONST, dtypes.int32, arg=offset)
    return UOp(Ops.STORE, dtypes.void, src=(buffer, offset_uop, data))


def amx_tdpbf16ps(c_tile: UOp, a_tile: UOp, b_tile: UOp) -> UOp:
    """
    AMX tdpbf16ps operation: C[16x16 FP32] += A[16x32 BF16] * B[16x32 BF16]
    
    Assembly equivalent: tdpbf16ps tmm0, tmm1, tmm2
    
    This performs:
    FOR m := 0 TO 15
        FOR k := 0 TO 15
            FOR n := 0 TO 15
                c[m][n] += FP32(a[m][2*k+0]) * FP32(b[k][2*n+0])
                c[m][n] += FP32(a[m][2*k+1]) * FP32(b[k][2*n+1])
    
    Args:
        c_tile: Accumulator tile (16x16 FP32)
        a_tile: Left matrix tile (16x32 BF16)
        b_tile: Right matrix tile (16x32 BF16)
    
    Returns:
        UOp representing updated C tile
    """
    # Matrix multiply: A * B
    matmul = UOp(Ops.WMMA, dtypes.float32, src=(a_tile, b_tile))
    
    # Accumulate: C += A * B
    result = UOp(Ops.ADD, dtypes.float32, src=(c_tile, matmul))
    
    return result


# ==============================================================================
# Example 1: Simple 16x16x32 MatMul (Single Tile)
# ==============================================================================

def amx_matmul_16x16x32_uop() -> UOp:
    """
    UOp representation of tdpbf16ps_N16_M16_K32_asm
    
    Assembly code:
    ```asm
    tdpbf16ps_N16_M16_K32_asm:
        mov         r10d, 64
        ldtilecfg   [config]
        tileloadd   tmm0, [rcx + r10]  ; load C
        tileloadd   tmm1, [rdx + r10]  ; load A
        tileloadd   tmm2, [r8 + r10]   ; load B
        tdpbf16ps   tmm0, tmm1, tmm2   ; C += A * B
        tilestored  [rcx + r10], tmm0  ; store C
        tilerelease
        ret
    ```
    
    Returns:
        UOp tree representing the complete operation
    """
    # Define buffers
    buf_c = create_amx_tile(0, rows=16, cols=16, dtype=dtypes.float32)
    buf_a = create_amx_tile(1, rows=16, cols=32, dtype=dtypes.bfloat16)
    buf_b = create_amx_tile(2, rows=16, cols=32, dtype=dtypes.bfloat16)
    
    # Stride = 64 bytes
    stride = 64
    
    # Load tiles
    c_loaded = amx_tile_load(buf_c, offset=stride)
    a_loaded = amx_tile_load(buf_a, offset=stride)
    b_loaded = amx_tile_load(buf_b, offset=stride)
    
    # Perform matrix multiply-accumulate
    c_updated = amx_tdpbf16ps(c_loaded, a_loaded, b_loaded)
    
    # Store result
    store_op = amx_tile_store(buf_c, c_updated, offset=stride)
    
    return store_op


# ==============================================================================
# Example 2: Multi-Tile 32x32x64 MatMul with Register Reuse
# ==============================================================================

def amx_matmul_32x32x64_uop() -> List[UOp]:
    """
    UOp representation of tdpbf16ps_N32_M32_K64_asm
    
    This performs 8 tile multiplications with optimal register reuse:
    - C is 32x32 = 2x2 tiles
    - A is 32x64 = 2x4 tiles (but only 2x2 needed for BF16)
    - B is 64x32 = 4x2 tiles (but only 2x2 needed for BF16)
    
    Operations:
    K=0:
        C[0][1] += A[0][1] * B[0][0]
        C[0][0] += A[0][0] * B[0][0]
        C[1][0] += A[0][0] * B[0][1]
        C[1][1] += A[0][1] * B[0][1]
    K=1:
        C[0][0] += A[1][0] * B[1][0]
        C[0][1] += A[1][1] * B[1][0]
        C[1][0] += A[1][0] * B[1][1]
        C[1][1] += A[1][1] * B[1][1]
    
    Returns:
        List of UOps for storing all result tiles
    """
    stride = 64
    
    # Define tile buffers - 2x2 grid for each matrix
    c_tiles = [[create_amx_tile(f"C_{i}_{j}", 16, 16, dtypes.float32) 
                for j in range(2)] for i in range(2)]
    a_tiles = [[create_amx_tile(f"A_{i}_{j}", 16, 32, dtypes.bfloat16) 
                for j in range(2)] for i in range(2)]
    b_tiles = [[create_amx_tile(f"B_{i}_{j}", 16, 32, dtypes.bfloat16) 
                for j in range(2)] for i in range(2)]
    
    # Track updated C tiles
    c_updated = [[None for _ in range(2)] for _ in range(2)]
    
    # K=0 iteration
    # Load A[0][1], B[0][0], C[0][1]
    a_01 = amx_tile_load(a_tiles[0][1], stride)
    b_00 = amx_tile_load(b_tiles[0][0], stride)
    c_01 = amx_tile_load(c_tiles[0][1], stride)
    
    # C[0][1] += A[0][1] * B[0][0]
    c_updated[0][1] = amx_tdpbf16ps(c_01, a_01, b_00)
    
    # Load A[0][0], C[0][0]
    a_00 = amx_tile_load(a_tiles[0][0], stride)
    c_00 = amx_tile_load(c_tiles[0][0], stride)
    
    # C[0][0] += A[0][0] * B[0][0] (reuse b_00)
    c_updated[0][0] = amx_tdpbf16ps(c_00, a_00, b_00)
    
    # Load B[0][1], C[1][0]
    b_01 = amx_tile_load(b_tiles[0][1], stride)
    c_10 = amx_tile_load(c_tiles[1][0], stride)
    
    # C[1][0] += A[0][0] * B[0][1] (reuse a_00)
    c_updated[1][0] = amx_tdpbf16ps(c_10, a_00, b_01)
    
    # Load C[1][1]
    c_11 = amx_tile_load(c_tiles[1][1], stride)
    
    # C[1][1] += A[0][1] * B[0][1] (reuse a_01, b_01)
    c_temp_11 = amx_tdpbf16ps(c_11, a_01, b_01)
    
    # K=1 iteration
    # Load A[1][0], B[1][0]
    a_10 = amx_tile_load(a_tiles[1][0], stride)
    b_10 = amx_tile_load(b_tiles[1][0], stride)
    
    # C[0][0] += A[1][0] * B[1][0]
    c_updated[0][0] = amx_tdpbf16ps(c_updated[0][0], a_10, b_10)
    
    # Load A[1][1]
    a_11 = amx_tile_load(a_tiles[1][1], stride)
    
    # C[0][1] += A[1][1] * B[1][0] (reuse b_10)
    c_updated[0][1] = amx_tdpbf16ps(c_updated[0][1], a_11, b_10)
    
    # Load B[1][1]
    b_11 = amx_tile_load(b_tiles[1][1], stride)
    
    # C[1][0] += A[1][0] * B[1][1] (reuse a_10)
    c_updated[1][0] = amx_tdpbf16ps(c_updated[1][0], a_10, b_11)
    
    # C[1][1] += A[1][1] * B[1][1] (reuse a_11, b_11)
    c_updated[1][1] = amx_tdpbf16ps(c_temp_11, a_11, b_11)
    
    # Store all C tiles
    stores = []
    for i in range(2):
        for j in range(2):
            store = amx_tile_store(c_tiles[i][j], c_updated[i][j], stride)
            stores.append(store)
    
    return stores


# ==============================================================================
# Example 3: Demonstrating UOp Singleton Pattern
# ==============================================================================

def demonstrate_singleton_pattern():
    """
    Demonstrate how UOp singleton pattern enables efficient comparison
    """
    print("=" * 70)
    print("Demonstrating UOp Singleton Pattern")
    print("=" * 70)
    
    # Same parameters = same instance
    const1 = UOp(Ops.CONST, dtypes.float32, arg=0.5)
    const2 = UOp(Ops.CONST, dtypes.float32, arg=0.5)
    
    print(f"\nconst1: {const1}")
    print(f"const2: {const2}")
    print(f"const1 == const2: {const1 == const2}")
    print(f"const1 is const2: {const1 is const2}  # Same object!")
    
    # Different parameters = different instances
    const3 = UOp(Ops.CONST, dtypes.float32, arg=1.0)
    print(f"\nconst3: {const3}")
    print(f"const1 == const3: {const1 == const3}")
    
    # Trees with same structure are equal
    buf1 = UOp(Ops.DEFINE_GLOBAL, dtypes.float32, arg=1)
    buf2 = UOp(Ops.DEFINE_GLOBAL, dtypes.float32, arg=1)
    
    tree1 = UOp(Ops.ADD, dtypes.float32, src=(const1, buf1))
    tree2 = UOp(Ops.ADD, dtypes.float32, src=(const2, buf2))
    
    print(f"\ntree1: {tree1}")
    print(f"tree2: {tree2}")
    print(f"tree1 == tree2: {tree1 == tree2}")
    print(f"tree1 is tree2: {tree1 is tree2}")


# ==============================================================================
# Example 4: Comparing Different MatMul Implementations
# ==============================================================================

def remove_loads(uop: UOp) -> Optional[UOp]:
    """
    Remove LOAD operations to compare computational patterns.
    This is useful for checking if two implementations perform
    the same computation, regardless of memory access patterns.
    """
    src = [remove_loads(u) for u in uop.src if u is not None]
    src = tuple([u for u in src if u is not None])
    
    if uop.op == Ops.LOAD:
        return None
    
    if not src and not uop.src:
        return uop
    
    return uop.replace(src=src)


def compare_matmul_implementations():
    """
    Compare two MatMul implementations to see if they're equivalent
    """
    print("\n" + "=" * 70)
    print("Comparing MatMul Implementations")
    print("=" * 70)
    
    # Create two implementations
    impl1 = amx_matmul_16x16x32_uop()
    impl2 = amx_matmul_16x16x32_uop()
    
    print(f"\nimpl1 == impl2: {impl1 == impl2}")
    print(f"impl1 is impl2: {impl1 is impl2}")
    
    # Remove memory operations to compare computation patterns
    pattern1 = remove_loads(impl1)
    pattern2 = remove_loads(impl2)
    
    print(f"\nAfter removing loads:")
    print(f"pattern1: {pattern1}")
    print(f"pattern1 == pattern2: {pattern1 == pattern2}")


# ==============================================================================
# Main Demo
# ==============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("AMX to Tinygrad UOp Conversion Examples")
    print("=" * 70)
    
    # Example 1: Simple matmul
    print("\n\n1. Simple 16x16x32 MatMul")
    print("-" * 70)
    kernel = amx_matmul_16x16x32_uop()
    print(f"Kernel UOp tree:\n{kernel}")
    
    # Example 2: Multi-tile matmul
    print("\n\n2. Multi-Tile 32x32x64 MatMul")
    print("-" * 70)
    kernels = amx_matmul_32x32x64_uop()
    print(f"Number of store operations: {len(kernels)}")
    print(f"First store operation:\n{kernels[0]}")
    
    # Example 3: Singleton pattern
    demonstrate_singleton_pattern()
    
    # Example 4: Compare implementations
    compare_matmul_implementations()
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. UOp singleton pattern enables efficient tree comparison")
    print("2. AMX operations map naturally to UOp primitives")
    print("3. Complex operations can be built compositionally")
    print("4. Optimization opportunities visible in UOp tree structure")
    print("\nSee tinygrad_uop_guide.md for detailed documentation.")


if __name__ == "__main__":
    main()

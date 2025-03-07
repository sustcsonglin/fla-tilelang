import argparse
import torch
import tilelang
import tilelang.language as T
from fla.ops.linear_attn import chunk_linear_attn
from functools import partial

def simple_linear_attn(batch, heads, seq_len, dim_qk, dim_v, BK, BV, BT):
    NK = T.ceildiv(dim_qk, BK)
    qk_shape = [batch, seq_len, heads, BK]
    v_shape = [batch, seq_len, heads, BV]
    o_shape = [NK, batch, seq_len, heads, BV] # we have to reduce the first dimension
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q: T.Buffer(qk_shape, dtype),
            K: T.Buffer(qk_shape, dtype),
            V: T.Buffer(v_shape, dtype),
            Output: T.Buffer(o_shape, dtype),
    ):
        with T.Kernel(heads, batch, T.ceildiv(dim_v, BV) * T.ceildiv(dim_qk, BK), threads=128) as (bx, by, bz):
            bk = bz % NK
            bv = bz // NK
            Q_shared = T.alloc_shared([BT, BK], dtype)
            K_shared = T.alloc_shared([BT, BK], dtype)
            K_local = T.alloc_fragment([BT, BK], dtype)
            K_local_trans = T.alloc_fragment([BK, BT], dtype)
            V_shared = T.alloc_shared([BT, BV], dtype)
            
            acc_o_local = T.alloc_fragment((BT, BV), accum_dtype)
            acc_o_shared = T.alloc_shared([BT, BV], dtype)

            # hidden state in register tiles, must be in fp32?
            acc_s_local = T.alloc_fragment((BK, BV), accum_dtype)
            acc_A_local = T.alloc_fragment((BT, BT), accum_dtype)
            acc_A_cast = T.alloc_fragment((BT, BT), dtype)

            acc_s_shared = T.alloc_shared((BK, BV), dtype, scope="shared")

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                acc_o_shared: tilelang.layout.make_swizzled_layout(acc_o_shared),
                acc_s_shared: tilelang.layout.make_swizzled_layout(acc_s_shared),
            })

            T.clear(acc_s_local)

            loop_range = T.ceildiv(seq_len, BT)
            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(K[by, k * BT:(k + 1) * BT, bx, bk * BK:(bk + 1) * BK], K_shared)
                T.copy(Q[by, k * BT:(k + 1) * BT, bx, bk * BK:(bk + 1) * BK], Q_shared)
                T.clear(acc_o_local)
                T.clear(acc_A_local)

                # SY: what does this policy mean?
                T.gemm(Q_shared, K_shared, acc_A_local, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, j in T.Parallel(BT, BT):
                    acc_A_local[i, j] = T.if_then_else(i >= j, acc_A_local[i, j], 0)

                T.copy(acc_A_local, acc_A_cast)
                T.copy(V[by, k * BT:(k + 1) * BT, bx, bv * BV:(bv + 1) * BV], V_shared)
                T.gemm(acc_A_cast, V_shared, acc_o_local, policy=T.GemmWarpPolicy.FullCol)
                T.copy(acc_s_local, acc_s_shared)
                T.gemm(Q_shared, acc_s_shared, acc_o_local, policy=T.GemmWarpPolicy.FullCol)
                T.copy(acc_o_local, acc_o_shared)
                T.copy(acc_o_shared, Output[bk, by, k * BT:(k + 1) * BT, bx, bv * BV:(bv + 1) * BV])
                
                # transpose k first because T.gemm does not have a good support for transposing the first operand according to the authors
                T.copy(K_shared, K_local)
                for i, j in T.Parallel(BK, BT):
                    K_local_trans[i, j] = K_local[j, i]
                T.gemm(K_local_trans, V_shared, acc_s_local, policy=T.GemmWarpPolicy.FullCol)

    return main


def ref_program(Q, K, V):
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K).tril()
    # qkm = qk * mask
    # r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o = torch.einsum('bhqk,bkhd->bqhd', qk, V)
    return o.to(dtype=torch.float16)

def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--h', type=int, default=10, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=4096, help='Context size')
    parser.add_argument('--dim_qk', type=int, default=256, help='Head dimension')
    parser.add_argument('--dim_v', type=int, default=256, help='Head dimension')
    args = parser.parse_args()
    BATCH, H, N_CTX, dim_qk, dim_v = args.batch, args.h, args.n_ctx, args.dim_qk, args.dim_v
    total_flops = 2.0 * BATCH * H * N_CTX * N_CTX * (dim_qk + dim_v)
    BLOCK_K = 128
    BLOCK_V = 128
    BLOCK_T = 64

    # TODO: auto padding
    assert dim_qk % BLOCK_K == 0
    assert dim_v % BLOCK_V == 0

    program = simple_linear_attn(BATCH, H, N_CTX, dim_qk, dim_v, BLOCK_K, BLOCK_V, BLOCK_T)
    mod, params = tilelang.lower(program)

    mod = tilelang.Profiler(mod, params, [3], tilelang.TensorSupplyType.Normal)

    ins = []
    for i in range(len(mod.params)):
        if i not in mod.result_idx:
            shape = [int(x) for x in mod.params[i].shape]
            ins.append(torch.randn(shape, device="cuda", dtype=torch.bfloat16))

    ref_outs = ref_program(*ins)
    torch.cuda.synchronize()
    lib_outs = mod.func(*ins)
    torch.cuda.synchronize()

    if isinstance(lib_outs, torch.Tensor):
        lib_outs = [lib_outs]
    if isinstance(ref_outs, torch.Tensor):
        ref_outs = [ref_outs]
    assert len(lib_outs) == len(ref_outs)

    for lhs, rhs in zip(lib_outs, ref_outs):
        print(get_err_ratio("Relative error: ", lhs.sum(0), rhs))
        print("If it is < 0.005, it is okayish")

    print("Caveat: TFLOPs might be misleading here, but the larger the faster..")

    latency = mod.do_bench(ref_program, n_warmup=10, n_repeat=10, profiler="torch")
    print("Ref: {:.2f} ms".format(latency))
    print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = mod.do_bench(mod, n_warmup=10, n_repeat=10, profiler="torch")
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    chunk_head_first = partial(chunk_linear_attn, head_first=False) 
    latency = mod.do_bench(chunk_head_first, n_warmup=10, n_repeat=10, profiler="torch")
    print("FLA: {:.2f} ms".format(latency))
    print("FLA: {:.2f} TFlops".format(total_flops / latency * 1e-9))

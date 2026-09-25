
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCHINDUCTOR_UNIQUE_KERNEL_NAMES'] = '1'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/workspace/vla_analysis/doc/gpu/compile_mode/inductor_debug'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/workspace/vla_analysis/doc/gpu/compile_mode/inductor_cache_fused_dump'
os.environ['TORCH_LOGS'] = 'output_code'
os.environ['TRITON_CACHE_DIR'] = '/workspace/vla_analysis/doc/gpu/compile_mode/inductor_cache_fused_dump/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.deterministic = False
torch._inductor.config.comprehensive_padding = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.11.0+cu128
# torch cuda version: 12.8
# torch git version: 70d99e998b4955e0049d13a98d77ae1b14db1f45


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Fri_Feb_21_20:23:50_PST_2025 
# Cuda compilation tools, release 12.8, V12.8.93 
# Build cuda_12.8.r12.8/compiler.35583870_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 5060 Ti : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1):
        view = torch.ops.aten.view.default(arg0_1, [1, -1]);  arg0_1 = None
        iota = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_2 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1 = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        full_default = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge = torch.ops.aten.ge.Scalar(unsqueeze_5, 0);  unsqueeze_5 = None
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(full_default, ge);  full_default = ge = None
        index = torch.ops.aten.index.Tensor(view, [unsqueeze_2, unsqueeze_8]);  view = unsqueeze_2 = unsqueeze_8 = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, index);  bitwise_and = index = None
        expand = torch.ops.aten.expand.default(bitwise_and_1, [1, -1, 1024, 1024]);  bitwise_and_1 = None
        clone = torch.ops.aten.clone.default(arg1_1, memory_format = torch.contiguous_format)
        convert_element_type = torch.ops.prims.convert_element_type.default(clone, torch.float32);  clone = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view_1 = torch.ops.aten.view.default(convert_element_type_1, [1024, 768])
        permute = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        addmm = torch.ops.aten.addmm.default(arg5_1, view_1, permute);  arg5_1 = view_1 = permute = None
        view_2 = torch.ops.aten.view.default(addmm, [1, 1024, 768]);  addmm = None
        view_3 = torch.ops.aten.view.default(view_2, [1, 1024, -1, 64]);  view_2 = None
        permute_1 = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4 = torch.ops.aten.view.default(convert_element_type_1, [1024, 768])
        permute_2 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg7_1, view_4, permute_2);  arg7_1 = view_4 = permute_2 = None
        view_5 = torch.ops.aten.view.default(addmm_1, [1, 1024, 768]);  addmm_1 = None
        view_6 = torch.ops.aten.view.default(view_5, [1, 1024, -1, 64]);  view_5 = None
        permute_3 = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        view_7 = torch.ops.aten.view.default(convert_element_type_1, [1024, 768]);  convert_element_type_1 = None
        permute_4 = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg9_1, view_7, permute_4);  arg9_1 = view_7 = permute_4 = None
        view_8 = torch.ops.aten.view.default(addmm_2, [1, 1024, 768]);  addmm_2 = None
        view_9 = torch.ops.aten.view.default(view_8, [1, 1024, -1, 64]);  view_8 = None
        permute_5 = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        full_default_1 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        expand_1 = torch.ops.aten.expand.default(where, [1, 12, 1024, 1024]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_1, False, scale = 0.125);  permute_1 = permute_3 = permute_5 = expand_1 = None
        getitem_2 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_10 = torch.ops.aten.view.default(permute_6, [1, 1024, -1]);  permute_6 = None
        view_11 = torch.ops.aten.view.default(view_10, [1024, 768]);  view_10 = None
        permute_7 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg11_1, view_11, permute_7);  arg11_1 = view_11 = permute_7 = None
        view_12 = torch.ops.aten.view.default(addmm_3, [1, 1024, 768]);  addmm_3 = None
        add_4 = torch.ops.aten.add.Tensor(arg1_1, view_12);  arg1_1 = view_12 = None
        clone_1 = torch.ops.aten.clone.default(add_4, memory_format = torch.contiguous_format)
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(clone_1, torch.float32);  clone_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_14, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_7);  convert_element_type_14 = getitem_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, arg13_1);  mul_3 = arg13_1 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_13 = torch.ops.aten.view.default(convert_element_type_15, [1024, 768]);  convert_element_type_15 = None
        permute_8 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg15_1, view_13, permute_8);  arg15_1 = view_13 = permute_8 = None
        view_14 = torch.ops.aten.view.default(addmm_4, [1, 1024, 3072]);  addmm_4 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_19, convert_element_type_19)
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, convert_element_type_19);  mul_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, 0.044715);  mul_5 = None
        add_7 = torch.ops.aten.add.Tensor(convert_element_type_19, mul_6);  mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.5);  convert_element_type_19 = None
        tanh = torch.ops.aten.tanh.default(mul_7);  mul_7 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, add_8);  mul_8 = add_8 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_15 = torch.ops.aten.view.default(convert_element_type_20, [1024, 3072]);  convert_element_type_20 = None
        permute_9 = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg17_1, view_15, permute_9);  arg17_1 = view_15 = permute_9 = None
        view_16 = torch.ops.aten.view.default(addmm_5, [1, 1024, 768]);  addmm_5 = None
        add_9 = torch.ops.aten.add.Tensor(add_4, view_16);  add_4 = view_16 = None
        clone_2 = torch.ops.aten.clone.default(add_9, memory_format = torch.contiguous_format)
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(clone_2, torch.float32);  clone_2 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_9);  convert_element_type_24 = getitem_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg18_1);  mul_10 = arg18_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_11, arg19_1);  mul_11 = arg19_1 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        view_17 = torch.ops.aten.view.default(convert_element_type_25, [1024, 768])
        permute_10 = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg21_1, view_17, permute_10);  arg21_1 = view_17 = permute_10 = None
        view_18 = torch.ops.aten.view.default(addmm_6, [1, 1024, 768]);  addmm_6 = None
        view_19 = torch.ops.aten.view.default(view_18, [1, 1024, -1, 64]);  view_18 = None
        permute_11 = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        view_20 = torch.ops.aten.view.default(convert_element_type_25, [1024, 768])
        permute_12 = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg23_1, view_20, permute_12);  arg23_1 = view_20 = permute_12 = None
        view_21 = torch.ops.aten.view.default(addmm_7, [1, 1024, 768]);  addmm_7 = None
        view_22 = torch.ops.aten.view.default(view_21, [1, 1024, -1, 64]);  view_21 = None
        permute_13 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        view_23 = torch.ops.aten.view.default(convert_element_type_25, [1024, 768]);  convert_element_type_25 = None
        permute_14 = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg25_1, view_23, permute_14);  arg25_1 = view_23 = permute_14 = None
        view_24 = torch.ops.aten.view.default(addmm_8, [1, 1024, 768]);  addmm_8 = None
        view_25 = torch.ops.aten.view.default(view_24, [1, 1024, -1, 64]);  view_24 = None
        permute_15 = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        full_default_3 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_2 = torch.ops.aten.expand.default(where_1, [1, 12, 1024, 1024]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_2, False, scale = 0.125);  permute_11 = permute_13 = permute_15 = expand_2 = None
        getitem_10 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_26 = torch.ops.aten.view.default(permute_16, [1, 1024, -1]);  permute_16 = None
        view_27 = torch.ops.aten.view.default(view_26, [1024, 768]);  view_26 = None
        permute_17 = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg27_1, view_27, permute_17);  arg27_1 = view_27 = permute_17 = None
        view_28 = torch.ops.aten.view.default(addmm_9, [1, 1024, 768]);  addmm_9 = None
        add_12 = torch.ops.aten.add.Tensor(add_9, view_28);  add_9 = view_28 = None
        clone_3 = torch.ops.aten.clone.default(add_12, memory_format = torch.contiguous_format)
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(clone_3, torch.float32);  clone_3 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_38, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_38, getitem_15);  convert_element_type_38 = getitem_15 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, arg28_1);  mul_12 = arg28_1 = None
        add_14 = torch.ops.aten.add.Tensor(mul_13, arg29_1);  mul_13 = arg29_1 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        view_29 = torch.ops.aten.view.default(convert_element_type_39, [1024, 768]);  convert_element_type_39 = None
        permute_18 = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg31_1, view_29, permute_18);  arg31_1 = view_29 = permute_18 = None
        view_30 = torch.ops.aten.view.default(addmm_10, [1, 1024, 3072]);  addmm_10 = None
        convert_element_type_43 = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_43, convert_element_type_43)
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, convert_element_type_43);  mul_14 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        add_15 = torch.ops.aten.add.Tensor(convert_element_type_43, mul_16);  mul_16 = None
        mul_17 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5);  convert_element_type_43 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_17);  mul_17 = None
        add_16 = torch.ops.aten.add.Tensor(tanh_1, 1);  tanh_1 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, add_16);  mul_18 = add_16 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        view_31 = torch.ops.aten.view.default(convert_element_type_44, [1024, 3072]);  convert_element_type_44 = None
        permute_19 = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg33_1, view_31, permute_19);  arg33_1 = view_31 = permute_19 = None
        view_32 = torch.ops.aten.view.default(addmm_11, [1, 1024, 768]);  addmm_11 = None
        add_17 = torch.ops.aten.add.Tensor(add_12, view_32);  add_12 = view_32 = None
        clone_4 = torch.ops.aten.clone.default(add_17, memory_format = torch.contiguous_format)
        convert_element_type_48 = torch.ops.prims.convert_element_type.default(clone_4, torch.float32);  clone_4 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_17);  convert_element_type_48 = getitem_17 = None
        mul_20 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_20, arg34_1);  mul_20 = arg34_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_21, arg35_1);  mul_21 = arg35_1 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        view_33 = torch.ops.aten.view.default(convert_element_type_49, [1024, 768])
        permute_20 = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg37_1, view_33, permute_20);  arg37_1 = view_33 = permute_20 = None
        view_34 = torch.ops.aten.view.default(addmm_12, [1, 1024, 768]);  addmm_12 = None
        view_35 = torch.ops.aten.view.default(view_34, [1, 1024, -1, 64]);  view_34 = None
        permute_21 = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_49, [1024, 768])
        permute_22 = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg39_1, view_36, permute_22);  arg39_1 = view_36 = permute_22 = None
        view_37 = torch.ops.aten.view.default(addmm_13, [1, 1024, 768]);  addmm_13 = None
        view_38 = torch.ops.aten.view.default(view_37, [1, 1024, -1, 64]);  view_37 = None
        permute_23 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        view_39 = torch.ops.aten.view.default(convert_element_type_49, [1024, 768]);  convert_element_type_49 = None
        permute_24 = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg41_1, view_39, permute_24);  arg41_1 = view_39 = permute_24 = None
        view_40 = torch.ops.aten.view.default(addmm_14, [1, 1024, 768]);  addmm_14 = None
        view_41 = torch.ops.aten.view.default(view_40, [1, 1024, -1, 64]);  view_40 = None
        permute_25 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        full_default_5 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(expand, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        expand_3 = torch.ops.aten.expand.default(where_2, [1, 12, 1024, 1024]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_3, False, scale = 0.125);  permute_21 = permute_23 = permute_25 = expand_3 = None
        getitem_18 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        view_42 = torch.ops.aten.view.default(permute_26, [1, 1024, -1]);  permute_26 = None
        view_43 = torch.ops.aten.view.default(view_42, [1024, 768]);  view_42 = None
        permute_27 = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg43_1, view_43, permute_27);  arg43_1 = view_43 = permute_27 = None
        view_44 = torch.ops.aten.view.default(addmm_15, [1, 1024, 768]);  addmm_15 = None
        add_20 = torch.ops.aten.add.Tensor(add_17, view_44);  add_17 = view_44 = None
        clone_5 = torch.ops.aten.clone.default(add_20, memory_format = torch.contiguous_format)
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(clone_5, torch.float32);  clone_5 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_5 = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_23);  convert_element_type_62 = getitem_23 = None
        mul_22 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, arg44_1);  mul_22 = arg44_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_23, arg45_1);  mul_23 = arg45_1 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        view_45 = torch.ops.aten.view.default(convert_element_type_63, [1024, 768]);  convert_element_type_63 = None
        permute_28 = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg47_1, view_45, permute_28);  arg47_1 = view_45 = permute_28 = None
        view_46 = torch.ops.aten.view.default(addmm_16, [1, 1024, 3072]);  addmm_16 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_67, convert_element_type_67)
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, convert_element_type_67);  mul_24 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, 0.044715);  mul_25 = None
        add_23 = torch.ops.aten.add.Tensor(convert_element_type_67, mul_26);  mul_26 = None
        mul_27 = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5);  convert_element_type_67 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_27);  mul_27 = None
        add_24 = torch.ops.aten.add.Tensor(tanh_2, 1);  tanh_2 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, add_24);  mul_28 = add_24 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_47 = torch.ops.aten.view.default(convert_element_type_68, [1024, 3072]);  convert_element_type_68 = None
        permute_29 = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg49_1, view_47, permute_29);  arg49_1 = view_47 = permute_29 = None
        view_48 = torch.ops.aten.view.default(addmm_17, [1, 1024, 768]);  addmm_17 = None
        add_25 = torch.ops.aten.add.Tensor(add_20, view_48);  add_20 = view_48 = None
        clone_6 = torch.ops.aten.clone.default(add_25, memory_format = torch.contiguous_format)
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(clone_6, torch.float32);  clone_6 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_6 = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_25);  convert_element_type_72 = getitem_25 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg50_1);  mul_30 = arg50_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_31, arg51_1);  mul_31 = arg51_1 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        view_49 = torch.ops.aten.view.default(convert_element_type_73, [1024, 768])
        permute_30 = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg53_1, view_49, permute_30);  arg53_1 = view_49 = permute_30 = None
        view_50 = torch.ops.aten.view.default(addmm_18, [1, 1024, 768]);  addmm_18 = None
        view_51 = torch.ops.aten.view.default(view_50, [1, 1024, -1, 64]);  view_50 = None
        permute_31 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        view_52 = torch.ops.aten.view.default(convert_element_type_73, [1024, 768])
        permute_32 = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg55_1, view_52, permute_32);  arg55_1 = view_52 = permute_32 = None
        view_53 = torch.ops.aten.view.default(addmm_19, [1, 1024, 768]);  addmm_19 = None
        view_54 = torch.ops.aten.view.default(view_53, [1, 1024, -1, 64]);  view_53 = None
        permute_33 = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        view_55 = torch.ops.aten.view.default(convert_element_type_73, [1024, 768]);  convert_element_type_73 = None
        permute_34 = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg57_1, view_55, permute_34);  arg57_1 = view_55 = permute_34 = None
        view_56 = torch.ops.aten.view.default(addmm_20, [1, 1024, 768]);  addmm_20 = None
        view_57 = torch.ops.aten.view.default(view_56, [1, 1024, -1, 64]);  view_56 = None
        permute_35 = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        full_default_7 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(expand, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        expand_4 = torch.ops.aten.expand.default(where_3, [1, 12, 1024, 1024]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_4, False, scale = 0.125);  permute_31 = permute_33 = permute_35 = expand_4 = None
        getitem_26 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        view_58 = torch.ops.aten.view.default(permute_36, [1, 1024, -1]);  permute_36 = None
        view_59 = torch.ops.aten.view.default(view_58, [1024, 768]);  view_58 = None
        permute_37 = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg59_1, view_59, permute_37);  arg59_1 = view_59 = permute_37 = None
        view_60 = torch.ops.aten.view.default(addmm_21, [1, 1024, 768]);  addmm_21 = None
        add_28 = torch.ops.aten.add.Tensor(add_25, view_60);  add_25 = view_60 = None
        clone_7 = torch.ops.aten.clone.default(add_28, memory_format = torch.contiguous_format)
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(clone_7, torch.float32);  clone_7 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_7 = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_31);  convert_element_type_86 = getitem_31 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg60_1);  mul_32 = arg60_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_33, arg61_1);  mul_33 = arg61_1 = None
        convert_element_type_87 = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None
        view_61 = torch.ops.aten.view.default(convert_element_type_87, [1024, 768]);  convert_element_type_87 = None
        permute_38 = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg63_1, view_61, permute_38);  arg63_1 = view_61 = permute_38 = None
        view_62 = torch.ops.aten.view.default(addmm_22, [1, 1024, 3072]);  addmm_22 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        mul_34 = torch.ops.aten.mul.Tensor(convert_element_type_91, convert_element_type_91)
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, convert_element_type_91);  mul_34 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, 0.044715);  mul_35 = None
        add_31 = torch.ops.aten.add.Tensor(convert_element_type_91, mul_36);  mul_36 = None
        mul_37 = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5);  convert_element_type_91 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_37);  mul_37 = None
        add_32 = torch.ops.aten.add.Tensor(tanh_3, 1);  tanh_3 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, add_32);  mul_38 = add_32 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        view_63 = torch.ops.aten.view.default(convert_element_type_92, [1024, 3072]);  convert_element_type_92 = None
        permute_39 = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg65_1, view_63, permute_39);  arg65_1 = view_63 = permute_39 = None
        view_64 = torch.ops.aten.view.default(addmm_23, [1, 1024, 768]);  addmm_23 = None
        add_33 = torch.ops.aten.add.Tensor(add_28, view_64);  add_28 = view_64 = None
        clone_8 = torch.ops.aten.clone.default(add_33, memory_format = torch.contiguous_format)
        convert_element_type_96 = torch.ops.prims.convert_element_type.default(clone_8, torch.float32);  clone_8 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_8 = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_33);  convert_element_type_96 = getitem_33 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg66_1);  mul_40 = arg66_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_41, arg67_1);  mul_41 = arg67_1 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        view_65 = torch.ops.aten.view.default(convert_element_type_97, [1024, 768])
        permute_40 = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg69_1, view_65, permute_40);  arg69_1 = view_65 = permute_40 = None
        view_66 = torch.ops.aten.view.default(addmm_24, [1, 1024, 768]);  addmm_24 = None
        view_67 = torch.ops.aten.view.default(view_66, [1, 1024, -1, 64]);  view_66 = None
        permute_41 = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
        view_68 = torch.ops.aten.view.default(convert_element_type_97, [1024, 768])
        permute_42 = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg71_1, view_68, permute_42);  arg71_1 = view_68 = permute_42 = None
        view_69 = torch.ops.aten.view.default(addmm_25, [1, 1024, 768]);  addmm_25 = None
        view_70 = torch.ops.aten.view.default(view_69, [1, 1024, -1, 64]);  view_69 = None
        permute_43 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        view_71 = torch.ops.aten.view.default(convert_element_type_97, [1024, 768]);  convert_element_type_97 = None
        permute_44 = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg73_1, view_71, permute_44);  arg73_1 = view_71 = permute_44 = None
        view_72 = torch.ops.aten.view.default(addmm_26, [1, 1024, 768]);  addmm_26 = None
        view_73 = torch.ops.aten.view.default(view_72, [1, 1024, -1, 64]);  view_72 = None
        permute_45 = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        full_default_9 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4 = torch.ops.aten.where.self(expand, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        expand_5 = torch.ops.aten.expand.default(where_4, [1, 12, 1024, 1024]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_5, False, scale = 0.125);  permute_41 = permute_43 = permute_45 = expand_5 = None
        getitem_34 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_74 = torch.ops.aten.view.default(permute_46, [1, 1024, -1]);  permute_46 = None
        view_75 = torch.ops.aten.view.default(view_74, [1024, 768]);  view_74 = None
        permute_47 = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg75_1, view_75, permute_47);  arg75_1 = view_75 = permute_47 = None
        view_76 = torch.ops.aten.view.default(addmm_27, [1, 1024, 768]);  addmm_27 = None
        add_36 = torch.ops.aten.add.Tensor(add_33, view_76);  add_33 = view_76 = None
        clone_9 = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
        convert_element_type_110 = torch.ops.prims.convert_element_type.default(clone_9, torch.float32);  clone_9 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_9 = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_39);  convert_element_type_110 = getitem_39 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg76_1);  mul_42 = arg76_1 = None
        add_38 = torch.ops.aten.add.Tensor(mul_43, arg77_1);  mul_43 = arg77_1 = None
        convert_element_type_111 = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None
        view_77 = torch.ops.aten.view.default(convert_element_type_111, [1024, 768]);  convert_element_type_111 = None
        permute_48 = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg79_1, view_77, permute_48);  arg79_1 = view_77 = permute_48 = None
        view_78 = torch.ops.aten.view.default(addmm_28, [1, 1024, 3072]);  addmm_28 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        mul_44 = torch.ops.aten.mul.Tensor(convert_element_type_115, convert_element_type_115)
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, convert_element_type_115);  mul_44 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 0.044715);  mul_45 = None
        add_39 = torch.ops.aten.add.Tensor(convert_element_type_115, mul_46);  mul_46 = None
        mul_47 = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5);  convert_element_type_115 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_47);  mul_47 = None
        add_40 = torch.ops.aten.add.Tensor(tanh_4, 1);  tanh_4 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, add_40);  mul_48 = add_40 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        view_79 = torch.ops.aten.view.default(convert_element_type_116, [1024, 3072]);  convert_element_type_116 = None
        permute_49 = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg81_1, view_79, permute_49);  arg81_1 = view_79 = permute_49 = None
        view_80 = torch.ops.aten.view.default(addmm_29, [1, 1024, 768]);  addmm_29 = None
        add_41 = torch.ops.aten.add.Tensor(add_36, view_80);  add_36 = view_80 = None
        clone_10 = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
        convert_element_type_120 = torch.ops.prims.convert_element_type.default(clone_10, torch.float32);  clone_10 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_10 = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_41);  convert_element_type_120 = getitem_41 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg82_1);  mul_50 = arg82_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_51, arg83_1);  mul_51 = arg83_1 = None
        convert_element_type_121 = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        view_81 = torch.ops.aten.view.default(convert_element_type_121, [1024, 768])
        permute_50 = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg85_1, view_81, permute_50);  arg85_1 = view_81 = permute_50 = None
        view_82 = torch.ops.aten.view.default(addmm_30, [1, 1024, 768]);  addmm_30 = None
        view_83 = torch.ops.aten.view.default(view_82, [1, 1024, -1, 64]);  view_82 = None
        permute_51 = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        view_84 = torch.ops.aten.view.default(convert_element_type_121, [1024, 768])
        permute_52 = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg87_1, view_84, permute_52);  arg87_1 = view_84 = permute_52 = None
        view_85 = torch.ops.aten.view.default(addmm_31, [1, 1024, 768]);  addmm_31 = None
        view_86 = torch.ops.aten.view.default(view_85, [1, 1024, -1, 64]);  view_85 = None
        permute_53 = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        view_87 = torch.ops.aten.view.default(convert_element_type_121, [1024, 768]);  convert_element_type_121 = None
        permute_54 = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg89_1, view_87, permute_54);  arg89_1 = view_87 = permute_54 = None
        view_88 = torch.ops.aten.view.default(addmm_32, [1, 1024, 768]);  addmm_32 = None
        view_89 = torch.ops.aten.view.default(view_88, [1, 1024, -1, 64]);  view_88 = None
        permute_55 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        full_default_11 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5 = torch.ops.aten.where.self(expand, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        expand_6 = torch.ops.aten.expand.default(where_5, [1, 12, 1024, 1024]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_6, False, scale = 0.125);  permute_51 = permute_53 = permute_55 = expand_6 = None
        getitem_42 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        view_90 = torch.ops.aten.view.default(permute_56, [1, 1024, -1]);  permute_56 = None
        view_91 = torch.ops.aten.view.default(view_90, [1024, 768]);  view_90 = None
        permute_57 = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg91_1, view_91, permute_57);  arg91_1 = view_91 = permute_57 = None
        view_92 = torch.ops.aten.view.default(addmm_33, [1, 1024, 768]);  addmm_33 = None
        add_44 = torch.ops.aten.add.Tensor(add_41, view_92);  add_41 = view_92 = None
        clone_11 = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
        convert_element_type_134 = torch.ops.prims.convert_element_type.default(clone_11, torch.float32);  clone_11 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_11 = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_47);  convert_element_type_134 = getitem_47 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, arg92_1);  mul_52 = arg92_1 = None
        add_46 = torch.ops.aten.add.Tensor(mul_53, arg93_1);  mul_53 = arg93_1 = None
        convert_element_type_135 = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        view_93 = torch.ops.aten.view.default(convert_element_type_135, [1024, 768]);  convert_element_type_135 = None
        permute_58 = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg95_1, view_93, permute_58);  arg95_1 = view_93 = permute_58 = None
        view_94 = torch.ops.aten.view.default(addmm_34, [1, 1024, 3072]);  addmm_34 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(view_94, torch.float32);  view_94 = None
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_139, convert_element_type_139)
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, convert_element_type_139);  mul_54 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, 0.044715);  mul_55 = None
        add_47 = torch.ops.aten.add.Tensor(convert_element_type_139, mul_56);  mul_56 = None
        mul_57 = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        mul_58 = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5);  convert_element_type_139 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_57);  mul_57 = None
        add_48 = torch.ops.aten.add.Tensor(tanh_5, 1);  tanh_5 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, add_48);  mul_58 = add_48 = None
        convert_element_type_140 = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        view_95 = torch.ops.aten.view.default(convert_element_type_140, [1024, 3072]);  convert_element_type_140 = None
        permute_59 = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg97_1, view_95, permute_59);  arg97_1 = view_95 = permute_59 = None
        view_96 = torch.ops.aten.view.default(addmm_35, [1, 1024, 768]);  addmm_35 = None
        add_49 = torch.ops.aten.add.Tensor(add_44, view_96);  add_44 = view_96 = None
        clone_12 = torch.ops.aten.clone.default(add_49, memory_format = torch.contiguous_format)
        convert_element_type_144 = torch.ops.prims.convert_element_type.default(clone_12, torch.float32);  clone_12 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_12 = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_49);  convert_element_type_144 = getitem_49 = None
        mul_60 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_61 = torch.ops.aten.mul.Tensor(mul_60, arg98_1);  mul_60 = arg98_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_61, arg99_1);  mul_61 = arg99_1 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        view_97 = torch.ops.aten.view.default(convert_element_type_145, [1024, 768])
        permute_60 = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg101_1, view_97, permute_60);  arg101_1 = view_97 = permute_60 = None
        view_98 = torch.ops.aten.view.default(addmm_36, [1, 1024, 768]);  addmm_36 = None
        view_99 = torch.ops.aten.view.default(view_98, [1, 1024, -1, 64]);  view_98 = None
        permute_61 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        view_100 = torch.ops.aten.view.default(convert_element_type_145, [1024, 768])
        permute_62 = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg103_1, view_100, permute_62);  arg103_1 = view_100 = permute_62 = None
        view_101 = torch.ops.aten.view.default(addmm_37, [1, 1024, 768]);  addmm_37 = None
        view_102 = torch.ops.aten.view.default(view_101, [1, 1024, -1, 64]);  view_101 = None
        permute_63 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        view_103 = torch.ops.aten.view.default(convert_element_type_145, [1024, 768]);  convert_element_type_145 = None
        permute_64 = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg105_1, view_103, permute_64);  arg105_1 = view_103 = permute_64 = None
        view_104 = torch.ops.aten.view.default(addmm_38, [1, 1024, 768]);  addmm_38 = None
        view_105 = torch.ops.aten.view.default(view_104, [1, 1024, -1, 64]);  view_104 = None
        permute_65 = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        full_default_13 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6 = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        expand_7 = torch.ops.aten.expand.default(where_6, [1, 12, 1024, 1024]);  where_6 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_63, permute_65, expand_7, False, scale = 0.125);  permute_61 = permute_63 = permute_65 = expand_7 = None
        getitem_50 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_66 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        view_106 = torch.ops.aten.view.default(permute_66, [1, 1024, -1]);  permute_66 = None
        view_107 = torch.ops.aten.view.default(view_106, [1024, 768]);  view_106 = None
        permute_67 = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg107_1, view_107, permute_67);  arg107_1 = view_107 = permute_67 = None
        view_108 = torch.ops.aten.view.default(addmm_39, [1, 1024, 768]);  addmm_39 = None
        add_52 = torch.ops.aten.add.Tensor(add_49, view_108);  add_49 = view_108 = None
        clone_13 = torch.ops.aten.clone.default(add_52, memory_format = torch.contiguous_format)
        convert_element_type_158 = torch.ops.prims.convert_element_type.default(clone_13, torch.float32);  clone_13 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_158, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_13[0]
        getitem_55 = var_mean_13[1];  var_mean_13 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_13 = torch.ops.aten.sub.Tensor(convert_element_type_158, getitem_55);  convert_element_type_158 = getitem_55 = None
        mul_62 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_62, arg108_1);  mul_62 = arg108_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_63, arg109_1);  mul_63 = arg109_1 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None
        view_109 = torch.ops.aten.view.default(convert_element_type_159, [1024, 768]);  convert_element_type_159 = None
        permute_68 = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg111_1, view_109, permute_68);  arg111_1 = view_109 = permute_68 = None
        view_110 = torch.ops.aten.view.default(addmm_40, [1, 1024, 3072]);  addmm_40 = None
        convert_element_type_163 = torch.ops.prims.convert_element_type.default(view_110, torch.float32);  view_110 = None
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_163, convert_element_type_163)
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, convert_element_type_163);  mul_64 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, 0.044715);  mul_65 = None
        add_55 = torch.ops.aten.add.Tensor(convert_element_type_163, mul_66);  mul_66 = None
        mul_67 = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        mul_68 = torch.ops.aten.mul.Tensor(convert_element_type_163, 0.5);  convert_element_type_163 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_67);  mul_67 = None
        add_56 = torch.ops.aten.add.Tensor(tanh_6, 1);  tanh_6 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, add_56);  mul_68 = add_56 = None
        convert_element_type_164 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        view_111 = torch.ops.aten.view.default(convert_element_type_164, [1024, 3072]);  convert_element_type_164 = None
        permute_69 = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg113_1, view_111, permute_69);  arg113_1 = view_111 = permute_69 = None
        view_112 = torch.ops.aten.view.default(addmm_41, [1, 1024, 768]);  addmm_41 = None
        add_57 = torch.ops.aten.add.Tensor(add_52, view_112);  add_52 = view_112 = None
        clone_14 = torch.ops.aten.clone.default(add_57, memory_format = torch.contiguous_format)
        convert_element_type_168 = torch.ops.prims.convert_element_type.default(clone_14, torch.float32);  clone_14 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_168, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_14[0]
        getitem_57 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_14 = torch.ops.aten.sub.Tensor(convert_element_type_168, getitem_57);  convert_element_type_168 = getitem_57 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, arg114_1);  mul_70 = arg114_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_71, arg115_1);  mul_71 = arg115_1 = None
        convert_element_type_169 = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        view_113 = torch.ops.aten.view.default(convert_element_type_169, [1024, 768])
        permute_70 = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg117_1, view_113, permute_70);  arg117_1 = view_113 = permute_70 = None
        view_114 = torch.ops.aten.view.default(addmm_42, [1, 1024, 768]);  addmm_42 = None
        view_115 = torch.ops.aten.view.default(view_114, [1, 1024, -1, 64]);  view_114 = None
        permute_71 = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        view_116 = torch.ops.aten.view.default(convert_element_type_169, [1024, 768])
        permute_72 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg119_1, view_116, permute_72);  arg119_1 = view_116 = permute_72 = None
        view_117 = torch.ops.aten.view.default(addmm_43, [1, 1024, 768]);  addmm_43 = None
        view_118 = torch.ops.aten.view.default(view_117, [1, 1024, -1, 64]);  view_117 = None
        permute_73 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        view_119 = torch.ops.aten.view.default(convert_element_type_169, [1024, 768]);  convert_element_type_169 = None
        permute_74 = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg121_1, view_119, permute_74);  arg121_1 = view_119 = permute_74 = None
        view_120 = torch.ops.aten.view.default(addmm_44, [1, 1024, 768]);  addmm_44 = None
        view_121 = torch.ops.aten.view.default(view_120, [1, 1024, -1, 64]);  view_120 = None
        permute_75 = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        full_default_15 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(expand, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        expand_8 = torch.ops.aten.expand.default(where_7, [1, 12, 1024, 1024]);  where_7 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_73, permute_75, expand_8, False, scale = 0.125);  permute_71 = permute_73 = permute_75 = expand_8 = None
        getitem_58 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_76 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        view_122 = torch.ops.aten.view.default(permute_76, [1, 1024, -1]);  permute_76 = None
        view_123 = torch.ops.aten.view.default(view_122, [1024, 768]);  view_122 = None
        permute_77 = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg123_1, view_123, permute_77);  arg123_1 = view_123 = permute_77 = None
        view_124 = torch.ops.aten.view.default(addmm_45, [1, 1024, 768]);  addmm_45 = None
        add_60 = torch.ops.aten.add.Tensor(add_57, view_124);  add_57 = view_124 = None
        clone_15 = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(clone_15, torch.float32);  clone_15 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_182, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_15[0]
        getitem_63 = var_mean_15[1];  var_mean_15 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_15 = torch.ops.aten.sub.Tensor(convert_element_type_182, getitem_63);  convert_element_type_182 = getitem_63 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg124_1);  mul_72 = arg124_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_73, arg125_1);  mul_73 = arg125_1 = None
        convert_element_type_183 = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None
        view_125 = torch.ops.aten.view.default(convert_element_type_183, [1024, 768]);  convert_element_type_183 = None
        permute_78 = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg127_1, view_125, permute_78);  arg127_1 = view_125 = permute_78 = None
        view_126 = torch.ops.aten.view.default(addmm_46, [1, 1024, 3072]);  addmm_46 = None
        convert_element_type_187 = torch.ops.prims.convert_element_type.default(view_126, torch.float32);  view_126 = None
        mul_74 = torch.ops.aten.mul.Tensor(convert_element_type_187, convert_element_type_187)
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, convert_element_type_187);  mul_74 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, 0.044715);  mul_75 = None
        add_63 = torch.ops.aten.add.Tensor(convert_element_type_187, mul_76);  mul_76 = None
        mul_77 = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.5);  convert_element_type_187 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_77);  mul_77 = None
        add_64 = torch.ops.aten.add.Tensor(tanh_7, 1);  tanh_7 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, add_64);  mul_78 = add_64 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        view_127 = torch.ops.aten.view.default(convert_element_type_188, [1024, 3072]);  convert_element_type_188 = None
        permute_79 = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg129_1, view_127, permute_79);  arg129_1 = view_127 = permute_79 = None
        view_128 = torch.ops.aten.view.default(addmm_47, [1, 1024, 768]);  addmm_47 = None
        add_65 = torch.ops.aten.add.Tensor(add_60, view_128);  add_60 = view_128 = None
        clone_16 = torch.ops.aten.clone.default(add_65, memory_format = torch.contiguous_format)
        convert_element_type_192 = torch.ops.prims.convert_element_type.default(clone_16, torch.float32);  clone_16 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_16[0]
        getitem_65 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_16 = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_65);  convert_element_type_192 = getitem_65 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg130_1);  mul_80 = arg130_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_81, arg131_1);  mul_81 = arg131_1 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        view_129 = torch.ops.aten.view.default(convert_element_type_193, [1024, 768])
        permute_80 = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg133_1, view_129, permute_80);  arg133_1 = view_129 = permute_80 = None
        view_130 = torch.ops.aten.view.default(addmm_48, [1, 1024, 768]);  addmm_48 = None
        view_131 = torch.ops.aten.view.default(view_130, [1, 1024, -1, 64]);  view_130 = None
        permute_81 = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        view_132 = torch.ops.aten.view.default(convert_element_type_193, [1024, 768])
        permute_82 = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg135_1, view_132, permute_82);  arg135_1 = view_132 = permute_82 = None
        view_133 = torch.ops.aten.view.default(addmm_49, [1, 1024, 768]);  addmm_49 = None
        view_134 = torch.ops.aten.view.default(view_133, [1, 1024, -1, 64]);  view_133 = None
        permute_83 = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        view_135 = torch.ops.aten.view.default(convert_element_type_193, [1024, 768]);  convert_element_type_193 = None
        permute_84 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg137_1, view_135, permute_84);  arg137_1 = view_135 = permute_84 = None
        view_136 = torch.ops.aten.view.default(addmm_50, [1, 1024, 768]);  addmm_50 = None
        view_137 = torch.ops.aten.view.default(view_136, [1, 1024, -1, 64]);  view_136 = None
        permute_85 = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        full_default_17 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8 = torch.ops.aten.where.self(expand, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        expand_9 = torch.ops.aten.expand.default(where_8, [1, 12, 1024, 1024]);  where_8 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_83, permute_85, expand_9, False, scale = 0.125);  permute_81 = permute_83 = permute_85 = expand_9 = None
        getitem_66 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_86 = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        view_138 = torch.ops.aten.view.default(permute_86, [1, 1024, -1]);  permute_86 = None
        view_139 = torch.ops.aten.view.default(view_138, [1024, 768]);  view_138 = None
        permute_87 = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg139_1, view_139, permute_87);  arg139_1 = view_139 = permute_87 = None
        view_140 = torch.ops.aten.view.default(addmm_51, [1, 1024, 768]);  addmm_51 = None
        add_68 = torch.ops.aten.add.Tensor(add_65, view_140);  add_65 = view_140 = None
        clone_17 = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
        convert_element_type_206 = torch.ops.prims.convert_element_type.default(clone_17, torch.float32);  clone_17 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_17[0]
        getitem_71 = var_mean_17[1];  var_mean_17 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_17 = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_71);  convert_element_type_206 = getitem_71 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg140_1);  mul_82 = arg140_1 = None
        add_70 = torch.ops.aten.add.Tensor(mul_83, arg141_1);  mul_83 = arg141_1 = None
        convert_element_type_207 = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None
        view_141 = torch.ops.aten.view.default(convert_element_type_207, [1024, 768]);  convert_element_type_207 = None
        permute_88 = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg143_1, view_141, permute_88);  arg143_1 = view_141 = permute_88 = None
        view_142 = torch.ops.aten.view.default(addmm_52, [1, 1024, 3072]);  addmm_52 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(view_142, torch.float32);  view_142 = None
        mul_84 = torch.ops.aten.mul.Tensor(convert_element_type_211, convert_element_type_211)
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, convert_element_type_211);  mul_84 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, 0.044715);  mul_85 = None
        add_71 = torch.ops.aten.add.Tensor(convert_element_type_211, mul_86);  mul_86 = None
        mul_87 = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.5);  convert_element_type_211 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_87);  mul_87 = None
        add_72 = torch.ops.aten.add.Tensor(tanh_8, 1);  tanh_8 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, add_72);  mul_88 = add_72 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        view_143 = torch.ops.aten.view.default(convert_element_type_212, [1024, 3072]);  convert_element_type_212 = None
        permute_89 = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg145_1, view_143, permute_89);  arg145_1 = view_143 = permute_89 = None
        view_144 = torch.ops.aten.view.default(addmm_53, [1, 1024, 768]);  addmm_53 = None
        add_73 = torch.ops.aten.add.Tensor(add_68, view_144);  add_68 = view_144 = None
        clone_18 = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
        convert_element_type_216 = torch.ops.prims.convert_element_type.default(clone_18, torch.float32);  clone_18 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_18[0]
        getitem_73 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_18 = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_73);  convert_element_type_216 = getitem_73 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg146_1);  mul_90 = arg146_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_91, arg147_1);  mul_91 = arg147_1 = None
        convert_element_type_217 = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None
        view_145 = torch.ops.aten.view.default(convert_element_type_217, [1024, 768])
        permute_90 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg149_1, view_145, permute_90);  arg149_1 = view_145 = permute_90 = None
        view_146 = torch.ops.aten.view.default(addmm_54, [1, 1024, 768]);  addmm_54 = None
        view_147 = torch.ops.aten.view.default(view_146, [1, 1024, -1, 64]);  view_146 = None
        permute_91 = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
        view_148 = torch.ops.aten.view.default(convert_element_type_217, [1024, 768])
        permute_92 = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg151_1, view_148, permute_92);  arg151_1 = view_148 = permute_92 = None
        view_149 = torch.ops.aten.view.default(addmm_55, [1, 1024, 768]);  addmm_55 = None
        view_150 = torch.ops.aten.view.default(view_149, [1, 1024, -1, 64]);  view_149 = None
        permute_93 = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        view_151 = torch.ops.aten.view.default(convert_element_type_217, [1024, 768]);  convert_element_type_217 = None
        permute_94 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg153_1, view_151, permute_94);  arg153_1 = view_151 = permute_94 = None
        view_152 = torch.ops.aten.view.default(addmm_56, [1, 1024, 768]);  addmm_56 = None
        view_153 = torch.ops.aten.view.default(view_152, [1, 1024, -1, 64]);  view_152 = None
        permute_95 = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        full_default_19 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9 = torch.ops.aten.where.self(expand, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        expand_10 = torch.ops.aten.expand.default(where_9, [1, 12, 1024, 1024]);  where_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_93, permute_95, expand_10, False, scale = 0.125);  permute_91 = permute_93 = permute_95 = expand_10 = None
        getitem_74 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_96 = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        view_154 = torch.ops.aten.view.default(permute_96, [1, 1024, -1]);  permute_96 = None
        view_155 = torch.ops.aten.view.default(view_154, [1024, 768]);  view_154 = None
        permute_97 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg155_1, view_155, permute_97);  arg155_1 = view_155 = permute_97 = None
        view_156 = torch.ops.aten.view.default(addmm_57, [1, 1024, 768]);  addmm_57 = None
        add_76 = torch.ops.aten.add.Tensor(add_73, view_156);  add_73 = view_156 = None
        clone_19 = torch.ops.aten.clone.default(add_76, memory_format = torch.contiguous_format)
        convert_element_type_230 = torch.ops.prims.convert_element_type.default(clone_19, torch.float32);  clone_19 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_19[0]
        getitem_79 = var_mean_19[1];  var_mean_19 = None
        add_77 = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_19 = torch.ops.aten.sub.Tensor(convert_element_type_230, getitem_79);  convert_element_type_230 = getitem_79 = None
        mul_92 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_93 = torch.ops.aten.mul.Tensor(mul_92, arg156_1);  mul_92 = arg156_1 = None
        add_78 = torch.ops.aten.add.Tensor(mul_93, arg157_1);  mul_93 = arg157_1 = None
        convert_element_type_231 = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        view_157 = torch.ops.aten.view.default(convert_element_type_231, [1024, 768]);  convert_element_type_231 = None
        permute_98 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg159_1, view_157, permute_98);  arg159_1 = view_157 = permute_98 = None
        view_158 = torch.ops.aten.view.default(addmm_58, [1, 1024, 3072]);  addmm_58 = None
        convert_element_type_235 = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_235, convert_element_type_235)
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, convert_element_type_235);  mul_94 = None
        mul_96 = torch.ops.aten.mul.Tensor(mul_95, 0.044715);  mul_95 = None
        add_79 = torch.ops.aten.add.Tensor(convert_element_type_235, mul_96);  mul_96 = None
        mul_97 = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        mul_98 = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.5);  convert_element_type_235 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_97);  mul_97 = None
        add_80 = torch.ops.aten.add.Tensor(tanh_9, 1);  tanh_9 = None
        mul_99 = torch.ops.aten.mul.Tensor(mul_98, add_80);  mul_98 = add_80 = None
        convert_element_type_236 = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        view_159 = torch.ops.aten.view.default(convert_element_type_236, [1024, 3072]);  convert_element_type_236 = None
        permute_99 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg161_1, view_159, permute_99);  arg161_1 = view_159 = permute_99 = None
        view_160 = torch.ops.aten.view.default(addmm_59, [1, 1024, 768]);  addmm_59 = None
        add_81 = torch.ops.aten.add.Tensor(add_76, view_160);  add_76 = view_160 = None
        clone_20 = torch.ops.aten.clone.default(add_81, memory_format = torch.contiguous_format)
        convert_element_type_240 = torch.ops.prims.convert_element_type.default(clone_20, torch.float32);  clone_20 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_20[0]
        getitem_81 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_20 = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_81);  convert_element_type_240 = getitem_81 = None
        mul_100 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, arg162_1);  mul_100 = arg162_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_101, arg163_1);  mul_101 = arg163_1 = None
        convert_element_type_241 = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        view_161 = torch.ops.aten.view.default(convert_element_type_241, [1024, 768])
        permute_100 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg165_1, view_161, permute_100);  arg165_1 = view_161 = permute_100 = None
        view_162 = torch.ops.aten.view.default(addmm_60, [1, 1024, 768]);  addmm_60 = None
        view_163 = torch.ops.aten.view.default(view_162, [1, 1024, -1, 64]);  view_162 = None
        permute_101 = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        view_164 = torch.ops.aten.view.default(convert_element_type_241, [1024, 768])
        permute_102 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg167_1, view_164, permute_102);  arg167_1 = view_164 = permute_102 = None
        view_165 = torch.ops.aten.view.default(addmm_61, [1, 1024, 768]);  addmm_61 = None
        view_166 = torch.ops.aten.view.default(view_165, [1, 1024, -1, 64]);  view_165 = None
        permute_103 = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        view_167 = torch.ops.aten.view.default(convert_element_type_241, [1024, 768]);  convert_element_type_241 = None
        permute_104 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg169_1, view_167, permute_104);  arg169_1 = view_167 = permute_104 = None
        view_168 = torch.ops.aten.view.default(addmm_62, [1, 1024, 768]);  addmm_62 = None
        view_169 = torch.ops.aten.view.default(view_168, [1, 1024, -1, 64]);  view_168 = None
        permute_105 = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        full_default_21 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10 = torch.ops.aten.where.self(expand, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        expand_11 = torch.ops.aten.expand.default(where_10, [1, 12, 1024, 1024]);  where_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_103, permute_105, expand_11, False, scale = 0.125);  permute_101 = permute_103 = permute_105 = expand_11 = None
        getitem_82 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_106 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_170 = torch.ops.aten.view.default(permute_106, [1, 1024, -1]);  permute_106 = None
        view_171 = torch.ops.aten.view.default(view_170, [1024, 768]);  view_170 = None
        permute_107 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg171_1, view_171, permute_107);  arg171_1 = view_171 = permute_107 = None
        view_172 = torch.ops.aten.view.default(addmm_63, [1, 1024, 768]);  addmm_63 = None
        add_84 = torch.ops.aten.add.Tensor(add_81, view_172);  add_81 = view_172 = None
        clone_21 = torch.ops.aten.clone.default(add_84, memory_format = torch.contiguous_format)
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(clone_21, torch.float32);  clone_21 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_254, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_21[0]
        getitem_87 = var_mean_21[1];  var_mean_21 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_21 = torch.ops.aten.sub.Tensor(convert_element_type_254, getitem_87);  convert_element_type_254 = getitem_87 = None
        mul_102 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_103 = torch.ops.aten.mul.Tensor(mul_102, arg172_1);  mul_102 = arg172_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_103, arg173_1);  mul_103 = arg173_1 = None
        convert_element_type_255 = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None
        view_173 = torch.ops.aten.view.default(convert_element_type_255, [1024, 768]);  convert_element_type_255 = None
        permute_108 = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg175_1, view_173, permute_108);  arg175_1 = view_173 = permute_108 = None
        view_174 = torch.ops.aten.view.default(addmm_64, [1, 1024, 3072]);  addmm_64 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(view_174, torch.float32);  view_174 = None
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_259, convert_element_type_259)
        mul_105 = torch.ops.aten.mul.Tensor(mul_104, convert_element_type_259);  mul_104 = None
        mul_106 = torch.ops.aten.mul.Tensor(mul_105, 0.044715);  mul_105 = None
        add_87 = torch.ops.aten.add.Tensor(convert_element_type_259, mul_106);  mul_106 = None
        mul_107 = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        mul_108 = torch.ops.aten.mul.Tensor(convert_element_type_259, 0.5);  convert_element_type_259 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_107);  mul_107 = None
        add_88 = torch.ops.aten.add.Tensor(tanh_10, 1);  tanh_10 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, add_88);  mul_108 = add_88 = None
        convert_element_type_260 = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_175 = torch.ops.aten.view.default(convert_element_type_260, [1024, 3072]);  convert_element_type_260 = None
        permute_109 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg177_1, view_175, permute_109);  arg177_1 = view_175 = permute_109 = None
        view_176 = torch.ops.aten.view.default(addmm_65, [1, 1024, 768]);  addmm_65 = None
        add_89 = torch.ops.aten.add.Tensor(add_84, view_176);  add_84 = view_176 = None
        clone_22 = torch.ops.aten.clone.default(add_89, memory_format = torch.contiguous_format)
        convert_element_type_264 = torch.ops.prims.convert_element_type.default(clone_22, torch.float32);  clone_22 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_264, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_22[0]
        getitem_89 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_22 = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_89);  convert_element_type_264 = getitem_89 = None
        mul_110 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_110, arg178_1);  mul_110 = arg178_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_111, arg179_1);  mul_111 = arg179_1 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        view_177 = torch.ops.aten.view.default(convert_element_type_265, [1024, 768])
        permute_110 = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg181_1, view_177, permute_110);  arg181_1 = view_177 = permute_110 = None
        view_178 = torch.ops.aten.view.default(addmm_66, [1, 1024, 768]);  addmm_66 = None
        view_179 = torch.ops.aten.view.default(view_178, [1, 1024, -1, 64]);  view_178 = None
        permute_111 = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
        view_180 = torch.ops.aten.view.default(convert_element_type_265, [1024, 768])
        permute_112 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg183_1, view_180, permute_112);  arg183_1 = view_180 = permute_112 = None
        view_181 = torch.ops.aten.view.default(addmm_67, [1, 1024, 768]);  addmm_67 = None
        view_182 = torch.ops.aten.view.default(view_181, [1, 1024, -1, 64]);  view_181 = None
        permute_113 = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        view_183 = torch.ops.aten.view.default(convert_element_type_265, [1024, 768]);  convert_element_type_265 = None
        permute_114 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg185_1, view_183, permute_114);  arg185_1 = view_183 = permute_114 = None
        view_184 = torch.ops.aten.view.default(addmm_68, [1, 1024, 768]);  addmm_68 = None
        view_185 = torch.ops.aten.view.default(view_184, [1, 1024, -1, 64]);  view_184 = None
        permute_115 = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        full_default_23 = torch.ops.aten.full.default([], -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11 = torch.ops.aten.where.self(expand, full_default_24, full_default_23);  expand = full_default_24 = full_default_23 = None
        expand_12 = torch.ops.aten.expand.default(where_11, [1, 12, 1024, 1024]);  where_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_113, permute_115, expand_12, False, scale = 0.125);  permute_111 = permute_113 = permute_115 = expand_12 = None
        getitem_90 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_116 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        view_186 = torch.ops.aten.view.default(permute_116, [1, 1024, -1]);  permute_116 = None
        view_187 = torch.ops.aten.view.default(view_186, [1024, 768]);  view_186 = None
        permute_117 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg187_1, view_187, permute_117);  arg187_1 = view_187 = permute_117 = None
        view_188 = torch.ops.aten.view.default(addmm_69, [1, 1024, 768]);  addmm_69 = None
        add_92 = torch.ops.aten.add.Tensor(add_89, view_188);  add_89 = view_188 = None
        clone_23 = torch.ops.aten.clone.default(add_92, memory_format = torch.contiguous_format)
        convert_element_type_278 = torch.ops.prims.convert_element_type.default(clone_23, torch.float32);  clone_23 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_278, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_23[0]
        getitem_95 = var_mean_23[1];  var_mean_23 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_23 = torch.ops.aten.sub.Tensor(convert_element_type_278, getitem_95);  convert_element_type_278 = getitem_95 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, arg188_1);  mul_112 = arg188_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_113, arg189_1);  mul_113 = arg189_1 = None
        convert_element_type_279 = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        view_189 = torch.ops.aten.view.default(convert_element_type_279, [1024, 768]);  convert_element_type_279 = None
        permute_118 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg191_1, view_189, permute_118);  arg191_1 = view_189 = permute_118 = None
        view_190 = torch.ops.aten.view.default(addmm_70, [1, 1024, 3072]);  addmm_70 = None
        convert_element_type_283 = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        mul_114 = torch.ops.aten.mul.Tensor(convert_element_type_283, convert_element_type_283)
        mul_115 = torch.ops.aten.mul.Tensor(mul_114, convert_element_type_283);  mul_114 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, 0.044715);  mul_115 = None
        add_95 = torch.ops.aten.add.Tensor(convert_element_type_283, mul_116);  mul_116 = None
        mul_117 = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_283, 0.5);  convert_element_type_283 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_117);  mul_117 = None
        add_96 = torch.ops.aten.add.Tensor(tanh_11, 1);  tanh_11 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, add_96);  mul_118 = add_96 = None
        convert_element_type_284 = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None
        view_191 = torch.ops.aten.view.default(convert_element_type_284, [1024, 3072]);  convert_element_type_284 = None
        permute_119 = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg193_1, view_191, permute_119);  arg193_1 = view_191 = permute_119 = None
        view_192 = torch.ops.aten.view.default(addmm_71, [1, 1024, 768]);  addmm_71 = None
        add_97 = torch.ops.aten.add.Tensor(add_92, view_192);  add_92 = view_192 = None
        clone_24 = torch.ops.aten.clone.default(add_97, memory_format = torch.contiguous_format);  add_97 = None
        convert_element_type_288 = torch.ops.prims.convert_element_type.default(clone_24, torch.float32);  clone_24 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_288, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_24[0]
        getitem_97 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_24 = torch.ops.aten.sub.Tensor(convert_element_type_288, getitem_97);  convert_element_type_288 = getitem_97 = None
        mul_120 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, arg194_1);  mul_120 = arg194_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_121, arg195_1);  mul_121 = arg195_1 = None
        convert_element_type_289 = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        return (convert_element_type_289,)

def load_args(reader):
    buf0 = reader.storage(None, 1024, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf0, (1, 32, 32), dtype=torch.bool, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1572864, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1, 1024, 768), (786432, 1, 1024), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf14, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf16, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf17, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf18, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf19, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf20, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf21, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf22, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf23, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf24, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf25, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf26, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf27, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf28, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf29, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf30, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf31, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf32, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf33, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf34, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf35, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf36, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf37, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf38, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf39, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf40, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf41, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf42, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf43, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf44, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf45, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf46, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf47, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf48, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf49, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf50, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf51, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf52, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf53, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf54, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf55, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf56, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf57, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf58, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf59, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf60, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf61, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf62, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf63, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf64, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf65, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf66, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf67, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf68, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf69, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf70, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf71, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf72, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf73, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf74, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf75, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf76, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf77, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf78, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf79, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf80, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf81, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf82, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf83, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf84, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf85, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf86, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf87, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf88, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf89, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf90, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf91, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf92, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf93, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf94, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf95, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf96, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf97, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf98, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf99, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf100, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf101, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf102, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf103, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf104, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf105, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf106, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf107, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf108, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf109, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf110, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf111, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf112, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf113, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf114, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf115, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf116, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf117, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf118, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf119, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf120, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf121, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf122, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf123, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf124, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf125, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf126, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf127, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf128, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf129, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf130, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf131, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf132, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf133, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf134, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf135, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf136, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf137, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf138, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf139, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf140, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf141, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf142, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf143, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf144, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf145, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf146, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf147, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf148, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf149, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf150, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf151, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf152, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf153, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf154, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf155, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf156, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf157, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf158, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf159, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf160, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf161, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf162, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf163, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf164, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf165, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf166, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf167, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf168, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf169, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf170, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf171, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf172, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf173, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf174, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf175, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf176, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf177, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf178, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf179, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf180, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf181, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf182, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf183, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf184, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf185, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf186, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf187, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf188, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf189, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf190, (3072, 768), dtype=torch.bfloat16, is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf191, (3072,), dtype=torch.bfloat16, is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf192, (768, 3072), dtype=torch.bfloat16, is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf193, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf194, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf195, (768,), dtype=torch.bfloat16, is_leaf=True)  # arg195_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)
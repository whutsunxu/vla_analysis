# Inductor post-fusion IR (sample)

Source: `TORCH_COMPILE_DEBUG=1` → `ir_post_fusion.txt` (Inductor scheduler after fusion).

| Item | Value |
|---|---|
| Graph sample | `model__4_inference_4.4` ≈ **FX/Inductor graph #5** (ViT encoder) |
| Raw IR | [`../../inductor_debug_samples/model__4_inference_4.4/ir_post_fusion.txt`](../../inductor_debug_samples/model__4_inference_4.4/ir_post_fusion.txt) (7483 lines) |
| Pre-fusion IR | [`../../inductor_debug_samples/model__4_inference_4.4/ir_pre_fusion.txt`](../../inductor_debug_samples/model__4_inference_4.4/ir_pre_fusion.txt) (6710 lines) |
| Codegen twin | [`../../inductor_debug_samples/model__4_inference_4.4/output_code.py`](../../inductor_debug_samples/model__4_inference_4.4/output_code.py) |
| Full 8-graph dumps | **Not in repo** (were under `inductor_debug/…/model__0…7` on the capture host; re-run `python src/smolvla_compile_fused_dump.py` on GPU) |

## Schedule nodes (post-fusion)

| # | Node | Kind | Note |
|---:|---|---|---|
| 1 | `op0_op1_op2` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 2 | `op0` | `SchedulerNode` |  |
| 3 | `op1` | `SchedulerNode` |  |
| 4 | `op2` | `SchedulerNode` |  |
| 5 | `op3_op4` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode) |
| 6 | `op3` | `SchedulerNode` |  |
| 7 | `op4` | `SchedulerNode` |  |
| 8 | `op6` | `SchedulerNode` |  |
| 9 | `op7` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 10 | `op8` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 11 | `op9` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 12 | `op10_op34_op56_op77` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 13 | `op10` | `SchedulerNode` |  |
| 14 | `op34` | `SchedulerNode` |  |
| 15 | `op56` | `SchedulerNode` |  |
| 16 | `op77` | `SchedulerNode` |  |
| 17 | `op11` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 18 | `op12` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 19 | `op16` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 20 | `op17_op18_op19` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 21 | `op17` | `SchedulerNode` |  |
| 22 | `op18` | `SchedulerNode` |  |
| 23 | `op19` | `SchedulerNode` |  |
| 24 | `op20_op21` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode) |
| 25 | `op20` | `SchedulerNode` |  |
| 26 | `op21` | `SchedulerNode` |  |
| 27 | `op23` | `SchedulerNode` |  |
| 28 | `op24` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 29 | `op25` | `SchedulerNode` |  |
| 30 | `op26` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 31 | `op27_op28_op30` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 32 | `op27` | `SchedulerNode` |  |
| 33 | `op28` | `SchedulerNode` |  |
| 34 | `op30` | `SchedulerNode` |  |
| 35 | `op31` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 36 | `op32` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 37 | `op33` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 38 | `op35` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 39 | `op36` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 40 | `op40` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 41 | `op41_op42_op44` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 42 | `op41` | `SchedulerNode` |  |
| 43 | `op42` | `SchedulerNode` |  |
| 44 | `op44` | `SchedulerNode` |  |
| 45 | `op45` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 46 | `op46` | `SchedulerNode` |  |
| 47 | `op47` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 48 | `op48_op49_op50_op52` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 49 | `op48` | `SchedulerNode` |  |
| 50 | `op49` | `SchedulerNode` |  |
| 51 | `op50` | `SchedulerNode` |  |
| 52 | `op52` | `SchedulerNode` |  |
| 53 | `op53` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 54 | `op54` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 55 | `op55` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 56 | `op57` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 57 | `op58` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 58 | `op62` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 59 | `op63_op64_op66` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 60 | `op63` | `SchedulerNode` |  |
| 61 | `op64` | `SchedulerNode` |  |
| 62 | `op66` | `SchedulerNode` |  |
| 63 | `op67` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 64 | `op68` | `SchedulerNode` |  |
| 65 | `op69` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 66 | `op70_op71_op73` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 67 | `op70` | `SchedulerNode` |  |
| 68 | `op71` | `SchedulerNode` |  |
| 69 | `op73` | `SchedulerNode` |  |
| 70 | `op74` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 71 | `op75` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 72 | `op76` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 73 | `op78` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 74 | `op79` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 75 | `op83` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 76 | `op84_op85_op87` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 77 | `op84` | `SchedulerNode` |  |
| 78 | `op85` | `SchedulerNode` |  |
| 79 | `op87` | `SchedulerNode` |  |
| 80 | `op88` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 81 | `op89` | `SchedulerNode` |  |
| 82 | `op90` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 83 | `op91_op92_op93_op95` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 84 | `op91` | `SchedulerNode` |  |
| 85 | `op92` | `SchedulerNode` |  |
| 86 | `op93` | `SchedulerNode` |  |
| 87 | `op95` | `SchedulerNode` |  |
| 88 | `op96` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 89 | `op97` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 90 | `op98` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 91 | `op99_op120_op142_op163` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 92 | `op99` | `SchedulerNode` |  |
| 93 | `op120` | `SchedulerNode` |  |
| 94 | `op142` | `SchedulerNode` |  |
| 95 | `op163` | `SchedulerNode` |  |
| 96 | `op100` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 97 | `op101` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 98 | `op105` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 99 | `op106_op107_op109` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 100 | `op106` | `SchedulerNode` |  |
| 101 | `op107` | `SchedulerNode` |  |
| 102 | `op109` | `SchedulerNode` |  |
| 103 | `op110` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 104 | `op111` | `SchedulerNode` |  |
| 105 | `op112` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 106 | `op113_op114_op116` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 107 | `op113` | `SchedulerNode` |  |
| 108 | `op114` | `SchedulerNode` |  |
| 109 | `op116` | `SchedulerNode` |  |
| 110 | `op117` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 111 | `op118` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 112 | `op119` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 113 | `op121` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 114 | `op122` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 115 | `op126` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 116 | `op127_op128_op130` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 117 | `op127` | `SchedulerNode` |  |
| 118 | `op128` | `SchedulerNode` |  |
| 119 | `op130` | `SchedulerNode` |  |
| 120 | `op131` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 121 | `op132` | `SchedulerNode` |  |
| 122 | `op133` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 123 | `op134_op135_op136_op138` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 124 | `op134` | `SchedulerNode` |  |
| 125 | `op135` | `SchedulerNode` |  |
| 126 | `op136` | `SchedulerNode` |  |
| 127 | `op138` | `SchedulerNode` |  |
| 128 | `op139` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 129 | `op140` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 130 | `op141` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 131 | `op143` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 132 | `op144` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 133 | `op148` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 134 | `op149_op150_op152` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 135 | `op149` | `SchedulerNode` |  |
| 136 | `op150` | `SchedulerNode` |  |
| 137 | `op152` | `SchedulerNode` |  |
| 138 | `op153` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 139 | `op154` | `SchedulerNode` |  |
| 140 | `op155` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 141 | `op156_op157_op159` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 142 | `op156` | `SchedulerNode` |  |
| 143 | `op157` | `SchedulerNode` |  |
| 144 | `op159` | `SchedulerNode` |  |
| 145 | `op160` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 146 | `op161` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 147 | `op162` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 148 | `op164` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 149 | `op165` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 150 | `op169` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 151 | `op170_op171_op173` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 152 | `op170` | `SchedulerNode` |  |
| 153 | `op171` | `SchedulerNode` |  |
| 154 | `op173` | `SchedulerNode` |  |
| 155 | `op174` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 156 | `op175` | `SchedulerNode` |  |
| 157 | `op176` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 158 | `op177_op178_op179_op181` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 159 | `op177` | `SchedulerNode` |  |
| 160 | `op178` | `SchedulerNode` |  |
| 161 | `op179` | `SchedulerNode` |  |
| 162 | `op181` | `SchedulerNode` |  |
| 163 | `op182` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 164 | `op183` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 165 | `op184` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 166 | `op185_op206_op228_op249` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 167 | `op185` | `SchedulerNode` |  |
| 168 | `op206` | `SchedulerNode` |  |
| 169 | `op228` | `SchedulerNode` |  |
| 170 | `op249` | `SchedulerNode` |  |
| 171 | `op186` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 172 | `op187` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 173 | `op191` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 174 | `op192_op193_op195` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 175 | `op192` | `SchedulerNode` |  |
| 176 | `op193` | `SchedulerNode` |  |
| 177 | `op195` | `SchedulerNode` |  |
| 178 | `op196` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 179 | `op197` | `SchedulerNode` |  |
| 180 | `op198` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 181 | `op199_op200_op202` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 182 | `op199` | `SchedulerNode` |  |
| 183 | `op200` | `SchedulerNode` |  |
| 184 | `op202` | `SchedulerNode` |  |
| 185 | `op203` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 186 | `op204` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 187 | `op205` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 188 | `op207` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 189 | `op208` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 190 | `op212` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 191 | `op213_op214_op216` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 192 | `op213` | `SchedulerNode` |  |
| 193 | `op214` | `SchedulerNode` |  |
| 194 | `op216` | `SchedulerNode` |  |
| 195 | `op217` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 196 | `op218` | `SchedulerNode` |  |
| 197 | `op219` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 198 | `op220_op221_op222_op224` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 199 | `op220` | `SchedulerNode` |  |
| 200 | `op221` | `SchedulerNode` |  |
| 201 | `op222` | `SchedulerNode` |  |
| 202 | `op224` | `SchedulerNode` |  |
| 203 | `op225` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 204 | `op226` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 205 | `op227` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 206 | `op229` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 207 | `op230` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 208 | `op234` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 209 | `op235_op236_op238` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 210 | `op235` | `SchedulerNode` |  |
| 211 | `op236` | `SchedulerNode` |  |
| 212 | `op238` | `SchedulerNode` |  |
| 213 | `op239` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 214 | `op240` | `SchedulerNode` |  |
| 215 | `op241` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 216 | `op242_op243_op245` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 217 | `op242` | `SchedulerNode` |  |
| 218 | `op243` | `SchedulerNode` |  |
| 219 | `op245` | `SchedulerNode` |  |
| 220 | `op246` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 221 | `op247` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 222 | `op248` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 223 | `op250` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 224 | `op251` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 225 | `op255` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 226 | `op256_op257_op259` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode) |
| 227 | `op256` | `SchedulerNode` |  |
| 228 | `op257` | `SchedulerNode` |  |
| 229 | `op259` | `SchedulerNode` |  |
| 230 | `op260` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 231 | `op261` | `SchedulerNode` |  |
| 232 | `op262` | `ExternKernelSchedulerNode` | extern (cuBLAS/cuDNN/…)  |
| 233 | `op263_op264_op265_op267` | `FusedSchedulerNode` | SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode) |
| 234 | `op263` | `SchedulerNode` |  |
| 235 | `op264` | `SchedulerNode` |  |
| 236 | `op265` | `SchedulerNode` |  |
| 237 | `op267` | `SchedulerNode` |  |

**Counts:** SchedulerNode=111, ExternKernelSchedulerNode=96, FusedSchedulerNode=30

## How to regenerate all 8 graphs

```bash
export SMOKE_DEVICE=cuda HF_HOME=/workspace/.hf_home
export SMOKE_COMPILE_MODE=reduce-overhead
python src/smolvla_compile_fused_dump.py
# → doc/gpu/compile_mode/inductor_debug/torch_compile_debug/run_*/torchinductor/model__N_*/ir_post_fusion.txt
```


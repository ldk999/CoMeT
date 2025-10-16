# TAS-MoE 实验代码设计指导文档（给代码助手的实现规范）

> 目的：将 **TAS-MoE**（TAPS：离线热感知映射 + TACS：在线热感知均衡）落地到 **CoMeT** 热仿真流程中，能跑通 Baseline/TAPS/TACS/TAS 四方案，并输出论文图表所需的统一 CSV 指标。本文档是**实现契约**：目录结构、数据模式、API、调度环路、与 CoMeT 的对接方式、测试与复现要求。

---

## 0. 术语与约定

- **逻辑层 trace**：与硬件位置解耦，仅描述模型计算/访存意图（MoE 路由、算子、张量读写）。
- **映射（TAPS）**：专家/张量 → 物理资源（core / DRAM bank）的离线分配。
- **覆盖补丁（TACS overlay）**：对“未来窗口”内待发事件的迁移/重路由说明（不改历史）。
- **物化（physicalization）**：将“逻辑层 trace + TAPS + TACS overlay”转为 **物理事件**（core、bank、hops、时间）喂给 CoMeT。
- **窗口（W）**：lookahead 时间窗（建议 2–5 ms 或 N 个 micro-batches），只允许在窗内改动未提交的事件。
- **提交边界（commit frontier）**：已注入 CoMeT 的事件不可回滚。
- **确定性**：给定相同随机种子、同一 `placement/overlay`，输出可完全复现。

> 设计要点：**调度改变的是“落点与路径”而非“语义”**。因此不直接篡改逻辑层，而是通过 **TAPS 映射 + TACS 覆盖补丁**，在**一次物化**时体现在物理事件中。

---

## 1. 目录结构与语言栈

```
comet-tasmoe/
  configs/
    arch_8x8_32x32x8.yaml        # 3D堆叠近存硬件、几何/材料/边界、DVFS/刷新参数
    workload_mixtral_8x7b.yaml   # MoE规模、top-k、序列、分布
    workload_deepseek_v2.yaml
    tas_params.yaml              # TAPS/TACS 超参
  workloads/
    generate_traces.py           # 逻辑层 trace 生成器
    trace_schema.md
  mapping/
    taps_mip.py                  # OR-Tools/PuLP MILP实现（可选）
    taps_heuristic.py            # 启发式（默认）
  runtime/
    tacs_runtime.py              # 在线均衡：overlay 生成
    physicalizer.py              # 物化器（窗口化流式）
    replica_state.py             # 专家副本/权重驻留状态
    overlay_types.py             # dataclass 定义
  adapters/
    comet_sink.py                # 写入 CoMeT 期望的事件格式/接口
    power_injector.py            # 事件→功耗注入（Compute/DRAM）
    dvfs_refresh_models.py       # T(℃)→刷新/带宽损失 & DVFS台阶
  metrics/
    aggregator.py                # 指标统计到统一CSV
    plots/                       # 可选：画图脚本
  scripts/
    run_all.py                   # 一键跑四方案
    run_single.py                # 跑一个工作负载+方案
  results/                       # 输出CSV
  traces/                        # 逻辑层与物理层事件缓存
```

- **语言**：Python 3.10+；必须可在纯 Python 环境下运行。MILP 依赖可选（若不可用，启用启发式）。
- **并行**：优先采用多进程/线程池在窗口粒度并发物化与仿真注入。

---

## 2. 配置文件模式（YAML）

### 2.1 硬件（`configs/arch_*.yaml`）
```yaml
mesh:
  cores: [8, 8]                # 8x8 core mesh
  noc: { topology: "mesh", width_bits: 256, link_latency_ns: 5 }
dram:
  banks: [32, 32, 8]           # x,y,layer
  page_kb: 8
geometry:
  macro_size_mm: [5.0, 2.0]    # 每宏缺省尺寸（项目基准）
  layers:                      # 逻辑/DRAM 层厚度、TIM、散热器、k/ρc/边界条件
    logic: { thickness_um: 50, k_wmk: 120 }
    dram:  { thickness_um: 50, k_wmk: 2 }
    tim:   { thickness_um: 20, k_wmk: 5 }
  cooling:
    type: "air"
    h_wmk2k: 10000
dvfs:
  domains: [{ cores: "all", steps_ghz: [2.0,1.8,1.6,1.2], trip_c: [80,85,90] }]
refresh:
  tref_base_us: 7.8
  tref_curve: [[40,1.0],[55,2.0],[70,4.0],[85,8.0]]  # 温度→倍数
```

### 2.2 工作负载（`configs/workload_*.yaml`）
```yaml
model: "mixtral-8x7b"
routing: { topk: 2, seed: 42, zipf_s: 1.2 }
sequence: { tokens: 2048, micro_batch: 32, steps: 500 }
operators:
  fc:   { flop_per_token: 2.5e9 }
  attn: { flop_per_token: 1.0e9, kv_bytes_per_token: 4096 }
tensors:
  weight_shard_mb: 64
```

### 2.3 TAS 超参（`configs/tas_params.yaml`）
```yaml
taps:
  objective: { w_peak: 1.0, w_var: 0.2, w_smooth: 0.1 }
  capacity:  { core_flops_tps: 5e12, bank_bw_gbps: 25 }
  replicate_hot_k: 1
tacs:
  window_ms: 3.0
  thresholds: { Thot: 80.0, Temg: 90.0 }
  q_target: 0.7                 # 目标负载比
  cost: { lambda_hops: 1.0, mu_noc: 2.0, eta_temp: 1.5 }
  max_parallel_migrations: 4
  min_residency_ms: 15
```

### 2.4 Trace 三层表示与“一次物化”

1) **逻辑层（scenario-agnostic）**：只描述模型事件，不含位置。  
2) **映射层（TAPS）**：把专家/张量放到物理资源（core/bank），与时间解耦。  
3) **覆盖层（TACS）**：对未来窗口的迁移/重路由补丁。  

> 物化器在窗口 `W` 内将三层合成“物理层事件”，并提交给 CoMeT。**已提交事件不可回滚**。

---

## 3. 数据模式（Schemas）

> 统一使用 CSV（UTF-8，逗号分隔，时间戳 ns）与 JSON（UTF-8，无注释）。

### 3.1 逻辑层（场景不可知）

**`traces/logical_compute.csv`**

| event_id | t_gen_ns | mb_id | token_id | experts_json | flops_json | op_type |
|---|---:|---:|---:|---|---|---|
| 1 | 0 | 0 | 0 | ["E17","E3"] | [1.2e9,9.5e8] | fc |

**`traces/logical_memory.csv`**

| event_id | tensor_id | access | bytes | locality_hint |
|---|---|---|---:|---|
| 42 | W_E17_0 | read | 1048576 | shard0 |

### 3.2 映射（TAPS 输出）

**`mapping/placement.json`**
```json
{
  "experts": { "E17": {"cores":[33], "policy":"single"}, "E3":{"cores":[5,37],"policy":"replicate"} },
  "tensors": { "W_E17_0":{"dram_bank":"B(12,7,3)"} }
}
```

### 3.3 运行时覆盖补丁（TACS 输出）

**`runtime/overlay_reroute.csv`**

| t_apply_ns | scope | from_core | to_core | mb_ids_json | reason | expected_gain |
|---:|---|---:|---:|---|---|---|
| 9000000 | mb | 33 | 21 | [12,13] | hot_core | 0.12 |

**`runtime/overlay_migrate.csv`**

| t_apply_ns | obj_type | obj_id | src | dst | size_bytes | cost |
|---:|---|---|---|---|---:|---:|
| 9000000 | weight | W_E17_0 | B(12,7,3) | B(10,6,3) | 67108864 | 2.3 |

### 3.4 物理层事件（喂给 CoMeT）

**`traces/compute_events.csv`**

| evt_id | t_start_ns | core_id | duration_cycles | flops | op | dvfs_domain | power_hint |
|---:|---:|---:|---:|---:|---|---|---|

**`traces/memory_events.csv`**

| evt_id | t_start_ns | src_core | dst_bank | bytes | rw | hops | noc_qos |
|---:|---:|---:|---|---:|---|---:|---|

### 3.5 副本状态（窗口推进所需）

**`runtime/replica_state.json`**
```json
{ "E3": { "cores":[5,37], "ts_created_ns": 1000000 } }
```

### 3.6 统一指标（输出）

**`results/<scenario>/<workload>.csv`**

| scenario | workload | T_peak_logic | T_peak_dram | T_avg | ips | lat_ms | bw_util | refresh_mhz | noc_load |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

---

## 4. 模块 API（函数签名与约束）

### 4.1 逻辑层 trace 生成（`workloads/generate_traces.py`）
```python
def generate_logical_traces(cfg_workload: dict, out_dir: str) -> tuple[str, str]:
    """
    Returns paths to logical_compute.csv and logical_memory.csv.
    Deterministic w.r.t cfg_workload and seed.
    """
```

### 4.2 TAPS（`mapping/taps_heuristic.py` 与可选 `taps_mip.py`）
```python
def build_placement(cfg_arch: dict, cfg_workload: dict, taps_params: dict,
                    H: "np.ndarray|None" = None) -> dict:
    """
    Returns placement.json object.
    Heuristic: (1) 估计专家热度 f_i；(2) 光谱/贪心在mesh上摊铺；
    (3) 热平滑正则（邻接核温度相近）；(4) bank 容量/带宽约束。
    """
```

### 4.3 TACS 运行时（`runtime/tacs_runtime.py`）
```python
from dataclasses import dataclass

@dataclass
class Telemetry:
    temps_core: list[float]           # ℃
    temps_bank: list[float]
    util_core: list[float]            # 0~1
    noc_load: float                   # 0~1
    ts_ns: int

def tacs_step(tele: Telemetry, placement: dict, state: "ReplicaState",
              params: dict, pending_window: "WindowView") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns overlay_reroute, overlay_migrate for next window.
    - 计算压力分数 S_c = [T_c-Thot]_+/(Temg-Thot) + γ [q_c-q_tgt]_+/q_tgt
    - 构建候选源/汇 A/B，限制最大迁移并发与最短驻留
    - 代价 cost = λ*hops + μ*NoĈ + η*ΔT̂
    - 选边并产出补丁（只影响未来窗口）
    """
```

### 4.4 物化器（`runtime/physicalizer.py`）
```python
def physicalize_window(win_idx: int,
                       logical_compute: "iterator",
                       logical_memory: "iterator",
                       placement: dict,
                       overlays: "Overlays",
                       state: "ReplicaState",
                       hw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply overlays for window, lower to physical events:
    - 填 core_id/bank/hops/dvfs_domain
    - 生成与迁移相关的额外 memory_events
    - 更新 ReplicaState
    """
```

### 4.5 CoMeT 对接（`adapters/comet_sink.py`）
```python
def submit_events_to_comet(compute_df, memory_df, cfg_arch) -> dict:
    """
    两种模式：
    (A) 文件模式：写 CoMeT 约定的事件CSV/JSON，然后调用其命令行
    (B) API模式：若存在Python接口，直接注入
    返回：本窗口内的温度（core/bank）、DVFS状态、刷新统计等快照
    """
```

### 4.6 功耗/热耦合（`adapters/power_injector.py` & `dvfs_refresh_models.py`）
```python
def compute_power_from_events(compute_df, memory_df, cfg_arch) -> dict:
    """
    P_logic = α_cmp * Σ FLOPs/t + leak(T_prev)
    P_mem   = Σ(events × per-access energy) + leak(T_prev)
    -> 提供给 CoMeT 的 p(t,space)
    """

def refresh_multiplier(temp_c: float) -> float: ...
def dvfs_step(temp_c: float, domain: str) -> float:  # 返回当前GHz
```

### 4.7 统一指标汇总（`metrics/aggregator.py`）
```python
def reduce_window_metrics(win_metrics: list[dict]) -> pd.DataFrame:
    """
    计算并拼接：T_peak/avg/var、ips/tokens_per_sec、lat、bw_util、refresh_mhz、noc_load
    输出 results/<scenario>/<workload>.csv
    """
```

### 4.8 脚本（`scripts/run_all.py`）
```python
def run_experiment(cfg_arch_path: str, cfg_workload_path: str, tas_params_path: str,
                   scenario: str, outdir: str) -> str:
    """
    scenario ∈ {baseline, taps, tacs, tas}
    - baseline: 不加载 placement/overlay，逻辑→默认映射
    - taps: placement 生效，overlay 空
    - tacs: baseline 映射 + 在线 overlay
    - tas: taps 映射 + 在线 overlay
    """
```

---

## 5. 调度环路（端到端时序）

```
for window in timeline:
  Lc,Lm = take_logical_events(window)                 # 生成器按窗口吐出
  OvR,OvM = (tacs_step(...) if scenario in {tacs,tas} else empty)
  C,M = physicalize_window(..., overlays=(OvR,OvM))
  metrics_win = submit_events_to_comet(C,M, cfg_arch) # 内含功耗注入与热求解
  log/emit metrics_win
reduce_window_metrics(...)
```

- **确定性**：同 `seed` 与相同 `placement/overlay` 输入 → 结果可重放。
- **不可回滚**：已提交窗口的事件不可被 TACS 修改。

---

## 6. 模型与代价（实现要点）

- **功耗**  
  `P_logic = α_cmp * (Σ FLOPs / Δt) + leak(T)`；`P_mem = Σ(access_energy) + leak(T)`  
- **热**  
  `T = H · p + T_amb`（CoMeT 解耦/求解，近似准稳态按窗口更新）
- **刷新/带宽**  
  `t_ref = tref_base × multiplier(T_dram)` → 有效带宽 `BW_eff = BW_raw × f(refresh)`  
- **DVFS**  
  阶梯式 GHz 随 `T_core` 触发降档；返回到 `T_cool` 后缓释回升（滞回）。

- **TACS 压力分数**  
  `S_c = [T_c - Thot]_+/(Temg - Thot) + γ [q_c - q_tgt]_+/q_tgt`  
- **迁移动作代价**  
  `cost(c→j) = λ·hops + μ·NoĈ(c→j) + η·ΔT̂_j`  
  约束：`max_parallel_migrations`、`min_residency_ms`、NoC 余量与 bank 带宽。

---

## 7. 边界条件与容错

- **依赖顺序**：迁移只作用于“未提交”的 micro-batches；不跨窗打乱读后写。
- **副本抖动**：`min_residency_ms` 与撤销冷却期；按窗口统计副本存活。
- **拥塞保护**：当 `noc_load` 过高暂缓迁移（或提高 `μ`）。
- **数值稳定**：窗口温度指数平滑（α=0.4~0.6）避免温度振荡。
- **缺省回退**：MILP 不可用 → 启发式；CoMeT API 不可用 → 文件模式。

---

## 8. 性能与扩展性要求

- **时间复杂度**：TACS 在每窗口仅考虑 top-K 热核与其邻域，边数 O(K·deg)。
- **并发物化**：窗口独立，使用进程池并发写入/解析事件。
- **I/O**：CSV 分段写、按窗口滚动落盘，避免巨文件。

---

## 9. 测试与验证（必须通过）

1. **单元测试**
   - `test_trace_determinism`：同 seed 逻辑层一致
   - `test_physicalizer_consistency`：不带 overlay 与空 overlay 等价
   - `test_replica_state_invariants`：最短驻留与撤销冷却生效
2. **集成测试**
   - Baseline 与 TAPS 在小工作负载（5 窗口）跑通，无异常 NaN/负温度
   - TACS 在手造热点（将专家集中到单核）时能降低 `T_peak_logic`
3. **数值 sanity**
   - 提升 `Thot` 会减少迁移次数
   - 提高 `μ`（NoC 代价）会降低 NoC 负载增量
4. **复现脚本**
   ```bash
   python scripts/run_all.py      --arch configs/arch_8x8_32x32x8.yaml      --workload configs/workload_mixtral_8x7b.yaml      --tas configs/tas_params.yaml      --out results/mixtral/
   ```

---

## 10. 输出与画图（可选）

- `metrics/plots/plot_temperature.py`：读多方案 CSV，绘制 `T_peak_{logic,dram}` 对比。
- `metrics/plots/plot_performance.py`：绘制 `ips` / `lat_ms` / `bw_util` 对比。
- 画图不作为通过条件，但 CSV 字段名与论文一致。

---

## 11. CoMeT 对接细节（两模式择一或都实现）

### 11.1 文件模式
- 写出 `compute_events.csv`、`memory_events.csv` 到 `traces/physical/`。
- 通过命令行调用 CoMeT，传入 `configs/arch_*.yaml` 与事件路径。
- 读取 CoMeT 返回的窗口温度分布（JSON/CSV），喂给 `aggregator` 与 `tacs_runtime`。

### 11.2 API 模式
- 若有 Python 接口：`comet.step(compute_batch, memory_batch, cfg)` → 返回温度/刷新增益。
- **幂等性**：每次 `step` 仅消费一个窗口的事件，禁止重入。

---

## 12. 实施顺序（MVP → 完整版）

1) 逻辑层 trace 生成器  
2) Baseline 物化与 CoMeT 打通（单窗口）  
3) 窗口化流水线 + 指标汇总  
4) 启发式 TAPS → 放热均匀化  
5) TACS overlay（迁移/重路由）  
6) MILP 版 TAPS（可选）与刷新/DVFS 曲线细化

---

## 13. 代码风格与质量门槛

- **类型注解** 强制，`mypy --strict` 通过。
- **日志**：`logging`，窗口级 INFO、调度决策 DEBUG；随机种子记录到 run-id。
- **错误处理**：I/O/外部工具异常必须被捕获并写入 `results/<run-id>/run.log`。
- **可重复**：所有随机性来源统一使用 `Random(seed)`。

---

## 14. 示例伪代码片段

```python
# scripts/run_single.py
def main():
    cfg_arch  = yaml.safe_load(open(args.arch))
    cfg_wl    = yaml.safe_load(open(args.workload))
    cfg_tas   = yaml.safe_load(open(args.tas))

    lc_path, lm_path = generate_logical_traces(cfg_wl, out_dir="traces/")
    placement = (build_placement(cfg_arch, cfg_wl, cfg_tas['taps'])
                 if args.scenario in {"taps","tas"} else default_placement(cfg_arch))
    state = ReplicaState()

    for win_idx in range(num_windows(cfg_wl, cfg_tas['tacs'])):
        tele = sense_last_window()  # from CoMeT output or init ambient
        overlays = (tacs_step(tele, placement, state, cfg_tas['tacs'], pending_view(win_idx))
                    if args.scenario in {"tacs","tas"} else empty_overlays())
        C,M = physicalize_window(win_idx, iter_logical(lc_path,lm_path),
                                 placement, overlays, state, cfg_arch)
        metrics = submit_events_to_comet(C,M,cfg_arch)
        aggregator.collect(metrics)
    aggregator.flush(out_csv)
```

---

## 15. 交付检查清单（DoD）

- [ ] 四方案均可运行，生成 `results/<scenario>/<workload>.csv`
- [ ] `T_peak_logic/T_peak_dram/ips` 字段存在且非空
- [ ] 产生 `run.log`，记录随机种子与配置快照
- [ ] `scripts/run_all.py` 一键跑通并产出 4 份 CSV
- [ ]（可选）`plots/` 能读 CSV 画出论文相似趋势图

---

**一句话指令给代码助手**  
按本文档目录与 API 契约搭好骨架；先跑通 Baseline→TAPS→TACS→TAS 的窗口化流水线与 CSV 指标，再细化 MILP、刷新/DVFS 与 NoC 代价模型。

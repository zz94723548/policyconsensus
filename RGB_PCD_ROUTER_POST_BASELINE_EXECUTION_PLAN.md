# RGB + PCD Router 基线验证后的详细执行方案

> 适用范围：RLBench MT4、RGB + Point Cloud、每个模态一个专家的两阶段模型。  
> 专家 checkpoint：`output/export_latest.ckpt`。  
> Router checkpoint：`output/router_latest.ckpt`。  
> 本文是 `COALITION_VALUE_CONFLICT_AWARE_MINIMAL_EXECUTION_PLAN.md` 的前置两模态验证方案，不替代后续 RGB+PCD+DINO 主线。

## 1. 当前结论与边界

当前 80 个 episode 的结果为：

| 方法 | Overall success rate |
|---|---:|
| RGB only | 0.5750 |
| PCD only | 0.5500 |
| RGB + PCD Router | 0.6625 |

Router 相比当前最佳单专家 RGB 提升 0.0875，即 8.75 个百分点；相比 PCD 提升 11.25 个百分点。checkpoint 参数检查也已确认：

- `export_latest.ckpt` 的 `ema_model` 与 `router_latest.ckpt` 的 RGB/PCD 编码器完全相同；
- 两个专家 UNet 的全部参数完全相同；
- normalizer 完全相同；
- 只有 `weight_predictor` 不同。

因此，可以确认提升不是由 Router 阶段更新专家造成的。但目前还不能得出“动态路由有效”的最终结论，因为尚未排除以下解释：

1. Router 只是学到了一个接近最优的固定比例；
2. 80 个 episode 的差异可能受测试方差影响；
3. 总体提升可能只来自一个 MT4 子任务；
4. 当前日志没有保存逐 episode 配对结果；
5. 还没有证明不同闭环状态确实需要不同权重。

接下来按 B0-B4 五个阶段执行：

```text
B0  固化当前基线与实验身份
 ↓
B1  固定权重扫描，确认 Router 是否超过最佳固定权重
 ↓
B2  固定随机源、重复性验收与80回合严格配对
 ↓
B3  环境状态恢复和分支一致性验收
 ↓
B4  Chunk / One-step-deviation Oracle
```

B3的状态恢复结果作为诊断和局限性分析，不再阻塞改版后的One-step-deviation Oracle。
只有One-step Oracle在独立episode上明显超过最佳固定权重及当前Router，才训练历史价值模型。

## 2. 统一实验约束

所有方法必须保持以下条件完全一致：

- 专家来源统一使用 `output/export_latest.ckpt` 的 `ema_model`；
- Router 使用 `output/router_latest.ckpt` 的 `ema_model`；
- 不更新 RGB encoder、PCD encoder、两个专家 UNet 和 normalizer；
- `n_obs_steps=2`、`n_action_steps=8`；
- diffusion inference steps 为 100；
- 环境最大长度为 250；
- MT4 四个任务循环均匀分配；
- 相同方法之间使用完全相同的 task、seed 和 episode 顺序；
- 开发集和最终测试集严格分开；
- 不根据最终测试结果重新选择固定权重或超参数。

MT4 任务顺序为：

1. `open_box`；
2. `toilet_seat_up`；
3. `open_drawer`；
4. `take_umbrella_out_of_umbrella_stand`。

## 3. B0：固化当前基线

### 3.1 保存实验身份

在任何后续修改前记录：

```bash
git rev-parse HEAD
sha256sum output/export_latest.ckpt output/router_latest.ckpt
```

把输出写入实验记录，不要改名覆盖原 checkpoint。建议实验目录结构：

```text
output/rgb_pcd_router_study/
├── metadata/
│   ├── checkpoints.sha256
│   ├── git_commit.txt
│   └── resolved_configs/
├── development/
├── confirmatory/
├── branch_validation/
└── oracle/
```

### 3.2 重跑三个当前基线

激活环境：

```bash
conda activate policy-consensus
```

RGB-only：

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/development/rgb_only_seed1000 \
  -n 80 \
  --num-videos 0 \
  --expert rgb \
  -d cuda:0
```

PCD-only：

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/development/pcd_only_seed1000 \
  -n 80 \
  --num-videos 0 \
  --expert pointcloud \
  -d cuda:0
```

Router：

```bash
python scripts/eval_policy.py sim \
  -c output/router_latest.ckpt \
  -o output/rgb_pcd_router_study/development/router_seed1000 \
  -n 80 \
  --num-videos 0 \
  -d cuda:0
```

### 3.3 B0 验收

- 三个结果与当前 0.575 / 0.55 / 0.6625 一致，或差异能由已记录的代码/随机性解释；
- `eval_log.json` 同时包含 Overall 和四个子任务成功率；
- 单专家模式实际只调用指定专家，不使用 sensor dropout 冒充单专家；
- 不把单专家模式中的 `avg_softmax_weight_*` 作为有效路由统计。

## 4. B1：实现并评测固定权重

### B1 代码状态

固定权重评测接口已经实现，五档 RLBench 实验由用户启动：

- `DiffusionUnetsPolicy.set_inference_fixed_weights()` 校验并保存最终融合概率；
- `--fixed-weights RGB_WEIGHT,PCD_WEIGHT` 已加入仿真评测 CLI；
- `--expert` 已统一复用固定权重机制；
- `--expert` 和 `--fixed-weights` 互斥；
- 固定权重在一次完整 diffusion sampling 中保持不变且不再 softmax；
- policy 输出 `router_logits` 和 `applied_weights`，二者语义分离；
- runner 聚合 `applied_weights`，日志字段改为 `TASK/avg_applied_weight_0/1`；
- 原 `avg_raw_weight_*` 和重复 softmax 得到的 `avg_softmax_weight_*` 不再生成。

### 4.1 为什么必须先做固定权重

令 RGB 权重为 `w`、PCD 权重为 `1-w`：

\[
\epsilon= w\epsilon_{RGB}+(1-w)\epsilon_{PCD}.
\]

候选集合固定为：

\[
\mathcal W=\{0,0.25,0.5,0.75,1\}.
\]

只有满足：

\[
SR_{Router} > \max_{w\in\mathcal W} SR_w
\]

才能初步说明状态相关 Router 比静态融合有价值。

### 4.2 已实现的推理接口

在 `DiffusionUnetsPolicy` 中增加仅用于推理的 `inference_fixed_weights`，其语义为最终融合概率，而不是 Router logits。要求：

- 接受长度等于专家数的非负数组；
- 权重和必须为 1，允许小的浮点误差；
- 固定权重存在时不调用 `weight_predictor` 决定融合比例；
- 一次完整 diffusion sampling 的 100 个去噪步骤使用同一个权重；
- 输出日志明确区分 `router_logits` 与 `applied_weights`；
- `--expert rgb` 等价于 `[1,0]`；
- `--expert pointcloud` 等价于 `[0,1]`。

`eval_policy_sim.py` 已增加：

```text
--fixed-weights RGB_WEIGHT,PCD_WEIGHT
```

`--test-start-seed` 和逐 episode 日志属于 B2，不包含在本次 B1 实现中。

互斥规则：

- `--expert` 与 `--fixed-weights` 不可同时使用；
- Router 正常评测时两者都不提供；
- 权重不是两个数、存在负数或和不为 1 时立即报错。

### 4.3 固定权重开发评测命令

以下命令现在可以直接执行。边界 0 和 1 已由单专家结果覆盖，但仍建议通过统一入口复核。

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/development/fixed_w000_seed1000 \
  -n 80 --num-videos 0 --fixed-weights 0,1 -d cuda:0
```

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/development/fixed_w025_seed1000 \
  -n 80 --num-videos 0 --fixed-weights 0.25,0.75 -d cuda:0
```

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/development/fixed_w050_seed1000 \
  -n 80 --num-videos 0 --fixed-weights 0.5,0.5 -d cuda:0
```

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/development/fixed_w075_seed1000 \
  -n 80 --num-videos 0 --fixed-weights 0.75,0.25 -d cuda:0
```

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/development/fixed_w100_seed1000 \
  -n 80 --num-videos 0 --fixed-weights 1,0 -d cuda:0
```

### 4.4 固定权重日志检查

固定 `[0.75,0.25]` 时，每个 episode 和每个重规划时刻的 `applied_weights` 都必须为：

```json
{"rgb": 0.75, "pointcloud": 0.25}
```

不得再次对该数组执行 softmax。因为：

```text
[0.75, 0.25] 是实际融合概率；
softmax([0.75, 0.25]) 不是实际融合概率。
```

### 4.5 B1 结果表

| 方法 | Overall | Open Box | Toilet Up | Open Drawer | Umbrella |
|---|---:|---:|---:|---:|---:|
| RGB only / w=1.00 | 0.5375 | 0.9500 | 0.3500 | 0.6000 | 0.2500 |
| w=0.75 | 0.5125 | 0.9000 | 0.5000 | 0.6500 | 0.0000 |
| **w=0.50（Best Fixed）** | **0.6375** | **0.9000** | **0.8500** | **0.7500** | **0.0500** |
| w=0.25 | 0.6000 | 1.0000 | 0.7000 | 0.6500 | 0.0500 |
| PCD only / w=0.00 | 0.5750 | 0.8500 | 0.7500 | 0.6500 | 0.0500 |
| Router | 0.6625 | 0.9500 | 0.6500 | 0.8000 | 0.2500 |

五档固定权重结果来自以下80回合评测日志，每个子任务20回合：

- `output/rgb_pcd_router_study/development/fixed_w000_seed1000/eval_log.json`；
- `output/rgb_pcd_router_study/development/fixed_w025_seed1000/eval_log.json`；
- `output/rgb_pcd_router_study/development/fixed_w050_seed1000/eval_log.json`；
- `output/rgb_pcd_router_study/development/fixed_w075_seed1000/eval_log.json`；
- `output/rgb_pcd_router_study/development/fixed_w100_seed1000/eval_log.json`。

Router 结果来自：

- Router：`output/eval_rgb_pcd_router/eval_log.json`。

Router 日志由修正 applied-weight 统计之前的代码生成，因此其中的
`avg_raw_weight_*` 和 `avg_softmax_weight_*` 不用于本表；成功率字段不受日志字段修正影响。

#### B1 初步结论

开发集上的最佳固定组合是 `RGB=0.5, PCD=0.5`，成功率为0.6375。Router
成功率为0.6625，仅高2.5个百分点，即80回合中相差2次成功。当前数据说明：

1. 固定权重组合不一定明显弱于动态Router；均匀固定组合已经接近Router；
2. 固定 `w=0.5` 在 `toilet_seat_up` 上为0.85，高于Router的0.65；
3. Router 在 `open_drawer` 和 `umbrella` 上更高，但逐任务每项只有20回合；
4. 现阶段不能声称动态权重显著优于固定权重，也不能声称两者等效；
5. 下一步必须把 `w=0.5` 固定为开发集选出的唯一Best Fixed，并在独立、严格配对的测试seed上与Router比较。

需要注意，统一固定权重入口重测得到的两个边界结果与此前 `--expert` 结果不同，例如
RGB-only从0.5750变为0.5375。这说明仅固定环境seed还不足以保证完全配对，diffusion
初始噪声等随机源也会带来方差。因此不能把当前2.5个百分点差异直接归因于动态路由。

### 4.6 B1 继续条件

- 若 Router 不高于最佳固定权重：先分析 Router 是否退化为近似常数，不进入长期价值模型；
- 若 Router 仅高 1-2 个百分点：进入 B2 扩大样本，暂不声称动态路由有效；
- 若 Router 在总体和多个任务上稳定高于最佳固定权重：进入 B2 配对确认。

## 5. B2：随机源固定、重复性验收与小规模严格配对

B2 不再直接运行每种方法400回合。当前目标是先建立可信的配对评测协议，证明同一方法可以重复，并确认 Router 与 Best Fixed 的比较没有被 diffusion 随机噪声主导。400回合确认性评测后移到 Oracle 和价值方法完成之后。

### B2 代码状态

5.1和5.2所需接口已经实现，尚未启动R0/R1/P0评测：

- CLI 已支持 `--test-start-seed`、`--diffusion-seed`、`--deterministic`、`--episode-log` 和 `--decision-log`；
- diffusion seed 由基础seed、task、environment seed和decision index稳定生成；
- batch中每个环境使用独立 `torch.Generator`，不受batch顺序和其他episode长度影响；
- 独立generator同时控制初始trajectory噪声和DDPM每个反向扩散步的随机噪声；
- parallel和non-parallel runner均可写入逐episode及逐decision JSONL；
- `eval_log.json` 额外记录测试起始seed、diffusion基础seed和deterministic状态。

轻量测试已验证相同逐样本seed产生完全相同的扩散结果，交换batch顺序后，同一seed对应的结果保持不变。完整RLBench重复性仍需通过下面的R0/R1命令验收。

### 5.1 B2 需要实现的命令行接口

在 `eval_policy_sim.py` 增加：

```text
--test-start-seed INTEGER
--diffusion-seed INTEGER
--deterministic
--episode-log PATH
--decision-log PATH       # 可选；调试和Oracle采集时启用
```

参数含义：

- `--test-start-seed`：设置第一个 RLBench episode seed，后续 episode 依次递增；
- `--diffusion-seed`：设置策略采样噪声的基础seed；
- `--deterministic`：固定 Python、NumPy、Torch CPU/CUDA 随机源，并启用可用的确定性后端设置；
- `--episode-log`：输出逐 episode 的 JSONL，而不仅是聚合 `eval_log.json`；
- `--decision-log`：输出每个重规划点的权重和采样seed，数据量较大，因此默认关闭。

仅调用 `torch.manual_seed()` 不足以形成严格配对。应为每个 `(task, environment_seed, decision_index)` 构造独立 diffusion seed，例如：

```python
decision_seed = stable_hash(
    diffusion_seed, task_name, environment_seed, decision_index
)
```

两种方法在相同 episode 和相同重规划序号必须获得相同的 diffusion 初始噪声。不能只维护一个全局 generator 并持续消耗，因为两种策略的 episode 长度不同后，后续随机数序列会错位。

推荐让策略支持以下等价接口之一：

```python
policy.set_inference_diffusion_seeds(per_env_seeds)
```

或：

```python
policy.predict_action(obs, diffusion_seeds=per_env_seeds)
```

并在 `conditional_sample()` 中为 batch 内每个环境分别生成初始噪声。seed 必须由 episode 身份和 decision index 决定，而不是由 batch 执行顺序决定。

### 5.2 逐 episode 和逐决策日志

`episodes.jsonl` 每行至少保存：

```json
{
  "method": "best_fixed_w050",
  "checkpoint": "output/export_latest.ckpt",
  "task": "open_box",
  "environment_seed": 2000,
  "diffusion_seed_base": 42000,
  "success": true,
  "episode_steps": 96,
  "num_replans": 12,
  "mean_applied_weights": [0.5, 0.5]
}
```

启用 `--decision-log` 时，`decisions.jsonl` 每行至少保存：

```json
{
  "task": "open_box",
  "environment_seed": 2000,
  "decision_index": 3,
  "decision_diffusion_seed": 8123456,
  "router_logits": [0.42, -0.18],
  "applied_weights": [0.6457, 0.3543]
}
```

固定权重的 `router_logits` 为 `null`。Router 每一步的正确统计是先对当前 logits 求 softmax，再沿时间平均 applied weights；固定权重不再执行softmax。

### 5.3 R0：Best Fixed 同方法重复性验收

先用20个 episode 做工程验收，每个 MT4 子任务5次。以下命令以 B2 接口实现完成为前提。

第一次：

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/repro/fixed_w050_run1 \
  -n 20 \
  --fixed-weights 0.5,0.5 \
  --test-start-seed 2000 \
  --diffusion-seed 42000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/repro/fixed_w050_run1/episodes.jsonl \
  --decision-log output/rgb_pcd_router_study/repro/fixed_w050_run1/decisions.jsonl \
  --num-videos 0 \
  -d cuda:0
```

第二次只更换输出目录：

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/repro/fixed_w050_run2 \
  -n 20 \
  --fixed-weights 0.5,0.5 \
  --test-start-seed 2000 \
  --diffusion-seed 42000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/repro/fixed_w050_run2/episodes.jsonl \
  --decision-log output/rgb_pcd_router_study/repro/fixed_w050_run2/decisions.jsonl \
  --num-videos 0 \
  -d cuda:0
```

比较键为 `(task, environment_seed)`。两次运行必须满足：

- episode 集合完全相同；
- success 标签100%一致；
- episode steps 和 replans 一致；
- decision diffusion seed 序列一致；
- applied weights 始终为 `[0.5,0.5]`；
- 动作若保存，最大误差为0或落在预先定义的数值容差内。

如果 success 仍不一致，不进入 Router 对比，先定位 RLBench、CUDA或状态初始化中的剩余随机源。

#### R0 首次运行记录

首次 `fixed_w050_run1/run2` 比较未通过：成功率分别为0.65和0.75，20个episode中有2个success标签不一致。虽然共同decision的diffusion seed与applied weights完全一致，但检查发现runner在带seed reset之后又执行了一次不带seed的reset；`RlbenchEnv.reset(seed=None)` 会调用 `np.random.seed(None)`，覆盖预期环境seed。该双重reset已修复，首次R0结果作废，必须用修复后的代码重新运行两次R0后再判断剩余物理非确定性。

修复reset后执行的 `fixed_w050_seedfix_run1/run2` 仍未通过：总体成功率分别为0.65和0.60，其中 `open_box` 分别为1.00和0.80。两个seedfix目录只保存了聚合 `eval_log.json`；JSONL路径没有同步改到新目录，因此无法定位具体不一致episode。这说明并行RLBench/CoppeliaSim仍存在剩余非确定性，下一步使用non-parallel runner重新执行R0。

Non-parallel R0第一次：

```bash
PARALLEL_RLBENCH=0 python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/repro/fixed_w050_nonparallel_run1 \
  -n 20 \
  --fixed-weights 0.5,0.5 \
  --test-start-seed 2000 \
  --diffusion-seed 42000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/repro/fixed_w050_nonparallel_run1/episodes.jsonl \
  --decision-log output/rgb_pcd_router_study/repro/fixed_w050_nonparallel_run1/decisions.jsonl \
  --num-videos 0 \
  -d cuda:0
```

Non-parallel R0第二次：

```bash
PARALLEL_RLBENCH=0 python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/repro/fixed_w050_nonparallel_run2 \
  -n 20 \
  --fixed-weights 0.5,0.5 \
  --test-start-seed 2000 \
  --diffusion-seed 42000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/repro/fixed_w050_nonparallel_run2/episodes.jsonl \
  --decision-log output/rgb_pcd_router_study/repro/fixed_w050_nonparallel_run2/decisions.jsonl \
  --num-videos 0 \
  -d cuda:0
```

### 5.4 R1：Router 同方法重复性验收

用同样的20个环境seed和diffusion seed运行两遍 Router：

```bash
python scripts/eval_policy.py sim \
  -c output/router_latest.ckpt \
  -o output/rgb_pcd_router_study/repro/router_run1 \
  -n 20 \
  --test-start-seed 2000 \
  --diffusion-seed 42000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/repro/router_run1/episodes.jsonl \
  --decision-log output/rgb_pcd_router_study/repro/router_run1/decisions.jsonl \
  --num-videos 0 \
  -d cuda:0
```

```bash
python scripts/eval_policy.py sim \
  -c output/router_latest.ckpt \
  -o output/rgb_pcd_router_study/repro/router_run2 \
  -n 20 \
  --test-start-seed 2000 \
  --diffusion-seed 42000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/repro/router_run2/episodes.jsonl \
  --decision-log output/rgb_pcd_router_study/repro/router_run2/decisions.jsonl \
  --num-videos 0 \
  -d cuda:0
```

验收条件与 R0 相同，并额外要求相同 decision 的 Router logits 和 applied weights 一致。

### 5.5 P0：Router 与 Best Fixed 的80回合严格配对

R0和R1通过后，使用新的测试seed `3000-3079`，避免复用选择 `w=0.5` 时的开发seed `1000-1079`。

Best Fixed：

```bash
python scripts/eval_policy.py sim \
  -c output/export_latest.ckpt \
  -o output/rgb_pcd_router_study/paired80/fixed_w050_seed3000 \
  -n 80 \
  --fixed-weights 0.5,0.5 \
  --test-start-seed 3000 \
  --diffusion-seed 52000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/paired80/fixed_w050_seed3000/episodes.jsonl \
  --num-videos 0 \
  -d cuda:0
```

Router：

```bash
python scripts/eval_policy.py sim \
  -c output/router_latest.ckpt \
  -o output/rgb_pcd_router_study/paired80/router_seed3000 \
  -n 80 \
  --test-start-seed 3000 \
  --diffusion-seed 52000 \
  --deterministic \
  --episode-log output/rgb_pcd_router_study/paired80/router_seed3000/episodes.jsonl \
  --decision-log output/rgb_pcd_router_study/paired80/router_seed3000/decisions.jsonl \
  --num-videos 0 \
  -d cuda:0
```

以 `(task, environment_seed)` 合并两个文件，至少报告：

- 两种方法总体和逐任务成功率；
- Router成功/Fixed失败的episode数；
- Router失败/Fixed成功的episode数；
- 两者都成功和都失败的episode数；
- Router每个任务的 applied-weight 均值、标准差和分位数；
- `Router - Fixed` 的配对成功率差。

80回合只用于判断趋势和检查配对协议，不用于声称统计显著性。此时不重新选择固定权重；无论结果如何，Best Fixed 仍锁定为开发集选出的 `[0.5,0.5]`。

### 5.6 B2 判定与后续路径

B2 的通过条件改为随机评测协议正确且能够量化噪声，不再要求相同方法的success标签完全一致：

1. 相同 `(task, environment_seed, decision_index)` 使用相同diffusion seed；
2. 日志中的 applied weights 与实际推理一致；
3. 重复运行能够给出方法自身的成功率波动范围；
4. P0能够形成80个按 `(task, environment_seed)` 对齐的episode；
5. 不再出现因全局generator消耗错位或无seed reset造成的伪差异。

B2通过后直接进入 B3 状态恢复和 B4 Oracle。原因是固定权重与Router的比较只能评价现有Router，不能回答“价值网络选择候选权重是否还有提升空间”。这个问题必须由同状态候选分支和Oracle回答。

400回合、paired bootstrap和McNemar确认性评测保留到价值方法完成后，统一比较 Best Fixed、Router、价值方法和Oracle，避免在尚未验证Oracle空间时消耗大量仿真预算。

## 6. B3：RLBench 状态恢复与分支噪声验收

B3不要求仿真完全确定。目标是测量恢复和动力学噪声，并判断候选权重效应是否明显大于该噪声。

### 6.1 已确认的RLBench能力与边界

当前依赖提供：

```python
task.get_state()
task.restore_state(state)
pyrep.set_configuration_tree(configuration_tree)
```

configuration tree覆盖对象相对位姿、关节值等配置。PyRep在恢复动态对象时会重置dynamic object，因此不能预先假设以下内容被精确恢复：

- 物体线速度和角速度；
- 瞬时接触力与接触求解器内部状态；
- 摩擦和约束求解的隐藏缓存；
- 所有任务自定义Python内部变量。

所以B3首先验收恢复能力，再决定论文术语：

- 恢复误差足够小：`restored-state paired branches`；
- 接触/速度无法可靠恢复：`matched-condition repeated branches`，不声称精确反事实。

### 6.2 需要实现的文件与接口

新增或扩展：

```text
modular_policy/env/rlbench/env.py
modular_policy/gymnasium_util/multistep_wrapper.py
modular_policy/workspace/validate_branch_state.py
scripts/validate_branch_state.py
```

以上接口与脚本现已实现。

环境接口：

```python
state = env.get_branch_state()
env.set_branch_state(state)
```

`branch_state`至少包含：

- task configuration tree及对象数量；
- robot arm与gripper configuration tree；
- 可访问的task object pose、linear velocity和angular velocity；
- `cur_step/done`；
- Python、NumPy、Torch CPU/CUDA RNG state；
- `MultiStepWrapper` 的obs/reward/done/truncate/info历史；
- 当前episode的task、environment seed和decision index；
- 当前policy历史、上一动作及上一权重；
- 当前decision diffusion seed。

恢复后先设置configuration tree和可设置速度，再恢复wrapper/policy/RNG状态；随后至少执行一个零推进或观测刷新步骤，检查是否出现物理跳变。

实际接口在configuration tree恢复后执行一次 `pyrep.step()`：CoppeliaSim重置动态对象后，必须推进一次才能重新获得joint-force等传感值。该推进造成的位姿、速度和观测变化计入B3恢复误差，不将其隐藏。

### 6.3 B3命令接口

下面命令现在可以直接执行。B3脚本自身只创建一个环境，不经过并行runner；命令保留 `PARALLEL_RLBENCH=0` 以明确实验约束。

公共参数含义：

- `--checkpoint`：冻结专家checkpoint；
- `--decision-index`：每个episode在哪个重规划时刻保存状态；
- `--weights`：候选RGB权重，PCD权重自动取 `1-w`；
- `--repeats`：同一个 `(state, weight)` 重复次数；
- `--prefix-actions`：每个分支执行多少个动作；
- `--continuation router`：动作前缀后统一切回冻结Router。

### 6.4 V0：状态接口冒烟测试

只运行1个episode，在decision 2保存、恢复，不做大规模分支：

```bash
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py smoke \
  --checkpoint output/router_latest.ckpt \
  --task mt4 \
  --episodes 1 \
  --test-start-seed 6000 \
  --diffusion-seed 62000 \
  --decision-index 2 \
  --output output/rgb_pcd_router_study/branch_validation/smoke
```

输出：

```text
restore_trials.jsonl
restore_summary.json
```

必须确认状态可以序列化、恢复调用不崩溃、任务和对象数量一致。

首次V0已通过接口冒烟验收。`open_box`静态对象配置精确恢复；恢复后的强制
`pyrep.step()` 使 `box_lid` 产生约0.52毫米最大位姿变化，并产生小幅速度差。
RGB观测存在小幅渲染差异。点云因FPS重采样不能使用逐索引MAE判断，验证脚本已增加
Chamfer distance、平均最近邻距离和点云质心误差，后续以这些集合指标为准。

### 6.5 V1：立即恢复误差

每个MT4任务取5个状态，共20个状态；每个状态立即恢复5次：

```bash
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py restore \
  --checkpoint output/router_latest.ckpt \
  --task mt4 \
  --episodes 20 \
  --test-start-seed 6100 \
  --diffusion-seed 63000 \
  --decision-index 4 \
  --repeats 5 \
  --output output/rgb_pcd_router_study/branch_validation/v1_restore
```

比较并报告：

- joint/gripper最大绝对误差；
- task object pose误差；
- 可获得的速度误差；
- RGB MAE与最大误差；
- point-cloud Chamfer distance或匹配点误差；
- 恢复后第一次仿真推进的状态跳变量。

V1已完成，共20个状态、每个恢复5次。机器人关节最大误差的95分位约
`7.07e-4 rad`；点云Chamfer L2的95分位约`4.32e-4`，同状态5次恢复的
Chamfer spread总体较小。主要异常来自 `take_umbrella_out_of_umbrella_stand`
的seed 6103：对象最大位姿偏移约6.82毫米，最大线速度差约0.151，最大角速度差
约2.49。这说明恢复过程可重复但存在动态状态的系统性恢复偏差，后续统一使用
`matched restored condition branches`，不称为原始状态的精确反事实。

### 6.6 V2：同状态、同权重重复噪声

小规模先验实验：每个任务5个状态，固定 `w=0.5`，每个状态重复5次：

先运行pilot，每个任务1个状态、每个状态重复3次，共12个完整分支：

```bash
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py repeat \
  --checkpoint output/router_latest.ckpt \
  --task mt4 \
  --episodes 4 \
  --test-start-seed 6200 \
  --diffusion-seed 64000 \
  --decision-index 4 \
  --weights 0.5 \
  --repeats 3 \
  --prefix-actions 8 \
  --continuation router \
  --output output/rgb_pcd_router_study/branch_validation/v2_same_weight_pilot \
  --device cuda:0
```

pilot能够完成且输出正确后，再执行下面20状态×5次的正式V2：

V2 pilot已完成（2026-11-10）。4个状态各重复3次，结果如下：

| task | seed | 3次终局成功 | continuation decisions | 成功方差 |
|---|---:|---|---|---:|
| open_box | 6200 | 1, 1, 1 | 18, 19, 19 | 0 |
| toilet_seat_up | 6201 | 0, 0, 0 | 27, 27, 27 | 0 |
| open_drawer | 6202 | 0, 0, 0 | 27, 27, 27 | 0 |
| take_umbrella_out_of_umbrella_stand | 6203 | 1, 1, 1 | 6, 6, 6 | 0 |

12个分支的终局标签完全一致；`open_box` 的完成时间存在1个决策周期波动，说明
恢复后的物理轨迹并非逐步完全相同，但本pilot的成功/失败标签稳定。由于这里只包含
4个状态，且只测试了一个权重，该结果只通过V2接口与小样本稳定性检查，不能单独证明
所有状态的噪声方差为零。`between_weight_effect.json` 在本阶段也没有比较意义。

考虑到正式V2的目的只是估计同权重噪声，而V3本身也会对每个权重重复3次，因此下一步
优先执行V3 pilot；它能够同时测量同权重噪声和不同权重效应。若V3 pilot出现明显的
同权重标签波动，再补跑下面的正式V2扩大噪声样本。

备用的正式V2命令：

```bash
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py repeat \
  --checkpoint output/router_latest.ckpt \
  --task mt4 \
  --episodes 20 \
  --test-start-seed 6200 \
  --diffusion-seed 64000 \
  --decision-index 4 \
  --weights 0.5 \
  --repeats 5 \
  --prefix-actions 8 \
  --continuation router \
  --output output/rgb_pcd_router_study/branch_validation/v2_same_weight
```

同一状态的5次重复使用相同decision diffusion seed，从而把测得变化主要归因于恢复与物理过程。输出至少包含：

```text
branches.jsonl
within_weight_variance.json
summary.json
```

### 6.7 V3：不同权重效应与噪声对比

在相同状态上测试五档权重。先用pilot确认信号，再以20个状态、每档3次作为正式实验；
只对不稳定状态追加重复：

先执行V3 pilot：4个状态、5档权重、每档重复3次，共60个分支。按V2 pilot速度估计
约需50至55分钟：

```bash
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py compare \
  --checkpoint output/router_latest.ckpt \
  --task mt4 \
  --episodes 4 \
  --test-start-seed 6200 \
  --diffusion-seed 64000 \
  --decision-index 4 \
  --weights 0,0.25,0.5,0.75,1 \
  --repeats 3 \
  --prefix-actions 8 \
  --continuation router \
  --output output/rgb_pcd_router_study/branch_validation/v3_weight_effect_pilot \
  --device cuda:0
```

该命令对与V2相同的4个恢复状态测试五档固定分支权重。每个分支先固定权重执行
`prefix-actions=8`个动作，随后恢复Router继续闭环至成功或超时。不要用V2中的
`between_weight_effect.json`判断权重效应；只有本命令包含多个权重，输出才有该含义。

V3 pilot完成后先检查：同一状态/权重的3次成功标签是否一致、不同权重之间是否出现
成功率差异，以及不同状态的最佳权重是否不同。满足这些条件后再扩展到20个状态。

V3 pilot已于2026-11-10完成，共60个分支，结果如下：

| task（seed） | w=0 | w=0.25 | w=0.5 | w=0.75 | w=1 | 当前最优 |
|---|---:|---:|---:|---:|---:|---|
| open_box（6200） | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 全部并列 |
| toilet_seat_up（6201） | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 全部并列失败 |
| open_drawer（6202） | 2/3 | 2/3 | 2/3 | 3/3 | 1/3 | w=0.75 |
| take_umbrella_out_of_umbrella_stand（6203） | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 全部并列 |

按全部12次分支计，各权重成功率分别为：`0.6667、0.6667、0.6667、0.7500、
0.5833`。但这不是独立episode总体成功率，而是4个恢复状态上的分支结果，不能与
此前80轮完整episode的成功率直接比较。

该pilot证明至少存在一个候选权重可能影响终局结果的状态：`open_drawer` 上权重间
success mean范围为0.667，`w=0.75`达到3/3。然而该状态在 `w=0、0.25、0.5、1`
下均出现同权重重复波动，样本成功方差为0.333。因此当前的权重效应尚未与恢复/物理
噪声充分分离，不能仅凭这个状态生成硬Oracle标签，也不能据此开始训练价值网络。

结论是V3 pilot通过“存在候选信号”的检查，但尚未通过正式B3统计验收。下一步扩展到
20个状态，并先采用每档3次重复，共300个分支；不立即增加到5次，以优先扩大状态覆盖。
按pilot速度估计约需3小时44分钟。若正式实验中仍有较多边界状态，再只对这些状态定向
增加重复次数，而不是把所有分支统一增加到5次。

正式V3命令：

```bash
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py compare \
  --checkpoint output/router_latest.ckpt \
  --task mt4 \
  --episodes 20 \
  --test-start-seed 6200 \
  --diffusion-seed 64000 \
  --decision-index 4 \
  --weights 0,0.25,0.5,0.75,1 \
  --repeats 3 \
  --prefix-actions 8 \
  --continuation router \
  --output output/rgb_pcd_router_study/branch_validation/v3_weight_effect_r3 \
  --device cuda:0
```

总分支数为：

```text
20 states × 5 weights × 3 repeats = 300 branches
```

输出包括 `branches.jsonl`、`within_weight_variance.json`、
`between_weight_effect.json` 和 `summary.json`。

正式结果出来后，根据within-weight variance识别不稳定状态。只对最优和次优候选差距
不大于同权重波动、或3次标签不一致的状态追加到5次；稳定状态无需重复计算。

#### 6.7.1 正式V3结果（2026-11-10）

正式V3已经完成：20个状态、5档权重、每档3次，共300个分支。总体统计如下：

| 指标 | 结果 |
|---|---:|
| 状态数 | 20 |
| 权重产生不同经验成功率的状态 | 5/20 |
| 具有唯一经验最优权重的状态 | 2/20 |
| 存在同权重重复波动的状态 | 5/20 |
| 所有权重均为3/3成功的状态 | 11/20 |
| 所有权重均为0/3失败的状态 | 4/20 |
| 非零同权重方差的状态-权重单元 | 17/100 |
| 全部单元平均样本方差 | 0.0567 |

五个权重敏感状态为：

| task（seed） | w=0 | w=0.25 | w=0.5 | w=0.75 | w=1 | 经验最优 |
|---|---:|---:|---:|---:|---:|---|
| open_drawer（6202） | 2/3 | 2/3 | 2/3 | 3/3 | 1/3 | 0.75 |
| open_drawer（6206） | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 0/0.25/0.5/1 |
| umbrella（6211） | 2/3 | 1/3 | 2/3 | 1/3 | 2/3 | 0/0.5/1 |
| umbrella（6215） | 1/3 | 1/3 | 0/3 | 1/3 | 2/3 | 1 |
| toilet_seat_up（6205） | 1/3 | 2/3 | 2/3 | 3/3 | 3/3 | 0.75/1 |

需要特别注意：这5个权重敏感状态也正好全部是存在同权重重复波动的状态。因此目前
观察到的between-weight effect尚未与恢复和物理噪声完全分离。其余15个状态没有提供
权重排序信息，其中11个对所有权重都成功，4个对所有权重都失败。

在全部分支上，五档权重的经验成功率为：

| RGB权重 | 经验成功率 |
|---:|---:|
| 0 | 0.7000 |
| 0.25 | 0.7000 |
| 0.5 | 0.7000 |
| 0.75 | 0.7167 |
| 1 | 0.7333 |

按状态事后选择样本均值最大的权重得到的经验Oracle为0.7667，相对同批数据中的最佳
固定权重 `w=1` 的0.7333仅高0.0333。这个差值不能解释为价值网络预期提升：经验
Oracle使用同一批小样本进行选择和评价，存在最大值选择偏差；它还是恢复状态上的分支
成功率，而不是完整episode成功率。

分任务的经验Oracle相对最佳固定权重差距为：`open_box=0`、`open_drawer=0.0667`、
`umbrella=0`、`toilet_seat_up=0`。当前观察到的Oracle增益几乎完全来自少数
`open_drawer` 边界状态。

因此正式V3尚不满足B3的严格恢复/反事实验收条件：虽然发现了候选权重信号，但没有证明多个状态
上的权重效应稳定大于同权重噪声，也没有得到足够多、可可靠区分的状态级最优权重标签。
当前不应直接开始训练价值网络。

另一个重要诊断是：seed 6202在pilot和正式V3两次独立运行中的15个终局标签逐项完全
相同，连各权重的重复序列都一致。这说明当前所谓的“重复波动”可能包含可复现的分支
执行顺序或未恢复仿真隐藏状态效应，并不满足独立同分布随机试验假设。当前循环固定采用
`weight -> repeat` 顺序，并在同一个模拟器实例中连续恢复同一snapshot；如果存在未纳入
snapshot的内部状态，后执行的分支可能受到前序分支影响。

因此下一步不能简单把这5个状态增加到7次并直接套用Wilson区间或Beta-Binomial后验，
否则会把系统性顺序偏差误当作统计置信度。应先完成以下顺序效应检查：

1. 支持 `--episode-indices 2,5,6,11,15`，只运行5个边界状态；
2. 支持候选权重和repeat顺序的确定性随机排列，并记录 `branch_order`；
3. 至少使用3个不同order seed重新运行；
4. 对关键状态增加“每个分支重建模拟器/重新捕获snapshot”的隔离对照；
5. 只有权重排序在不同执行顺序和隔离对照下仍稳定，才把结果视为Oracle信号；
6. 通过上述检查后，再将每个状态-权重扩展到至少7个近似独立样本，并生成软价值及置信度。

为避免继续运行原命令、重复当前的系统性模式，下一节先加入离散episode选择和顺序
随机化接口。每分支隔离成本更高，待顺序效应复测结果出来后再决定是否实现和执行。

#### 6.7.2 B3.1顺序效应复测命令

定向选择和顺序随机化接口现已实现：

- `--episode-indices 2,5,6,11,15`：保留原始episode索引、任务映射和环境seed，只运行
  五个边界状态；
- `--shuffle-branches`：分别打乱每个状态内的 `(weight, repeat)` 执行顺序；
- `--order-seed`：使顺序可复现；
- `branches.jsonl` 新增 `episode_index`、`branch_order`、`execution_index`、
  `order_seed`、`state_order_seed` 和 `branches_shuffled`。

依次执行三个order seed，每组75个分支：

```bash
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py compare \
  --checkpoint output/router_latest.ckpt \
  --task mt4 --episodes 20 --episode-indices 2,5,6,11,15 \
  --test-start-seed 6200 --diffusion-seed 64000 --decision-index 4 \
  --weights 0,0.25,0.5,0.75,1 --repeats 3 --prefix-actions 8 \
  --continuation router --shuffle-branches --order-seed 71001 \
  --output output/rgb_pcd_router_study/branch_validation/v3_order_71001 \
  --device cuda:0 && \
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py compare \
  --checkpoint output/router_latest.ckpt \
  --task mt4 --episodes 20 --episode-indices 2,5,6,11,15 \
  --test-start-seed 6200 --diffusion-seed 64000 --decision-index 4 \
  --weights 0,0.25,0.5,0.75,1 --repeats 3 --prefix-actions 8 \
  --continuation router --shuffle-branches --order-seed 71002 \
  --output output/rgb_pcd_router_study/branch_validation/v3_order_71002 \
  --device cuda:0 && \
PARALLEL_RLBENCH=0 python scripts/validate_branch_state.py compare \
  --checkpoint output/router_latest.ckpt \
  --task mt4 --episodes 20 --episode-indices 2,5,6,11,15 \
  --test-start-seed 6200 --diffusion-seed 64000 --decision-index 4 \
  --weights 0,0.25,0.5,0.75,1 --repeats 3 --prefix-actions 8 \
  --continuation router --shuffle-branches --order-seed 71003 \
  --output output/rgb_pcd_router_study/branch_validation/v3_order_71003 \
  --device cuda:0
```

三个命令用 `&&` 连接：前一组正常完成后才启动下一组。预计每组约71分钟，三组总计
约3小时33分钟。三组必须保持环境seed、diffusion seed和其他参数不变，只改变
`order-seed`，从而把差异归因于分支执行顺序。

#### 6.7.3 B3.1结果与结论

三组顺序复测已经完成（`order_seed=71001/71002/71003`）。结果显示执行顺序影响非常
强，不能把当前分支结果视为独立重复：

- `open_drawer` seed 6202：71001下五档权重均为3/3成功；71002下仍均为3/3；
  71003下则为RGB权重0的2/3、其余权重3/3。
- `open_drawer` seed 6206：71001、71002、71003均为大多3/3，但71001的0.75为
  2/3，说明边界仍会随顺序改变。
- `umbrella` seed 6211：71001下各权重为0/3至2/3不等；71002下全部0/3；
  71003下全部3/3。
- `umbrella` seed 6215：71001下只有少量成功；71002下全部0/3；71003下全部0/3。
- `toilet_seat_up` seed 6205：71001下大多数权重2/3；71002下权重结果在0/3和
  3/3之间跳变；71003下各权重为1/3至2/3。

例如 umbrella seed 6211 在三个顺序下整体从“部分成功”变成“全部失败”再变成
“全部成功”，这不是候选权重的小幅影响，而是分支执行顺序/模拟器隐藏状态造成的系统性
变化。因而：

1. 当前B3尚未通过；
2. 之前的within-weight variance不能解释为独立实验噪声；
3. 当前的经验Oracle和最佳权重不能用于训练价值网络；
4. 继续改变 `order-seed` 不能解决问题，只能反复采样同一个受污染的模拟器过程。

原计划要求先完成逐状态的精确分支隔离，再构造全轨迹Oracle；这一要求现在调整为诊断项，
不再作为价值网络路线的硬门槛。B2/B3的顺序污染说明当前恢复接口不能提供严格配对的
反事实，但它不否定在独立episode分布上估计期望单步价值的可行性。

### 6.8 改版目标：One-step-deviation Oracle

`Compose Your Policies!`的理论结论支持本项目的核心假设：在相同条件下，多个策略的
分布得分做凸组合，可以获得比单个策略更低的单步函数误差；在稳定性条件下，单步误差
改进可以传播到生成轨迹。因此本项目不再追求“每个扩散去噪步都重新预测权重”，也不
要求用全轨迹Oracle枚举所有未来权重。

本项目采用以下定义。给定重规划状态 (s_t)、候选权重 (w) 和固定 continuation
(pi_c)，只让候选权重影响当前一个动作chunk，随后始终由 (pi_c) 接管：

\[
Q^{\pi_c}_{1\text{-dev}}(s_t,w)
=\mathbb{E}[G(s_t;\text{one chunk with }w,\text{ then }\pi_c)].
\]

这不是完整最优Q函数，而是“相对于固定后续策略的一步偏离价值”。它回答的问题是：
在当前状态，当前动作chunk应该偏向RGB、PCD还是中间凸组合，才能让固定后续策略取得
更好的结果。

#### 6.8.1 推荐的标签协议

1. 用普通Router/RGB/PCD episode采集状态 (s_t)，不要求以后能逐位恢复同一CoppeliaSim
   状态；环境随机性被视为标签噪声。
2. 对每个状态评估五个候选权重 `0, 0.25, 0.5, 0.75, 1`，每个候选只执行一个
   `n_action_steps=8` 的动作chunk。
3. chunk结束后切换到冻结的 continuation。主实验使用 Router continuation；另加
   RGB-only 和 PCD-only continuation 作为控制变量。
4. 标签使用终局成功、折扣回报或二者组合，不强制生成单一硬最优权重。推荐保存
   `q_mean`、`q_std`、`n_samples`，并用归一化后的soft target表示候选偏好。
5. 由于环境不可完全配对，使用独立episode的样本均值和置信区间，不把一次分支的胜负
   宣称为确定性因果结论。

#### 6.8.2 价值网络形式

优先训练 (V(s,w)\rightarrow Q^{\pi_c}_{1\text{-dev}}(s,w))，而不是直接训练
`argmax` 分类器。输入为RGB/PCD编码和候选权重，输出标量价值及可选不确定性；部署时
在五个候选权重上前向5次并取最大值。这样既支持离散候选，也能扩展到连续权重。

训练目标可以使用Huber回归或带不确定性的高斯NLL；如果只训练偏好，则使用由
`q_mean/q_std`构造的softmax soft label。只有当最佳候选和次优候选的置信区间明显分离
时，才记录硬Oracle标签，否则保留并列或软标签。

#### 6.8.3 部署语义与论文表述

价值网络不参与扩散过程中的每个denoising timestep，也不需要每个环境仿真步调用。
它在一次策略重规划时选择当前动作chunk的组合权重；该chunk执行后，下一次重规划可以
再次调用，或者按实验设定切换到固定Router continuation。论文中应称为
`one-step-deviation value-guided composition`，不要称为完整轨迹Oracle或全局最优策略。

这条路线的必要假设是：

- 候选策略共享同一状态和动作空间，组合权重确实作用于当前chunk的score/action生成；
- continuation在候选比较期间固定，否则标签同时混入后续策略差异；
- 单步价值与任务成功之间存在可学习相关性；
- 训练和评测使用独立episode，报告均值、置信区间和任务级结果。

#### 6.8.4 最小可行实验

先不继续扩大B3恢复重复。直接构造四组对照：Router、最佳固定权重、one-step value
选择+Router continuation、one-step value选择+RGB/PCD continuation。若one-step value
在独立episode上超过最佳固定权重且开销只增加一次小型价值网络前向，即可支持价值网络
路线；B3的恢复误差作为局限性分析，而不是路线否决条件。

### 6.9 B3诊断指标与通过条件

对每个状态和权重估计：

\[
\bar Q(s,w)=\frac{1}{R}\sum_r G^{(r)}(s,w),
\qquad
\hat\sigma^2_{within}(s,w)
=\frac{1}{R-1}\sum_r(G^{(r)}-\bar Q)^2.
\]

再比较状态内最佳与次优候选差距：

\[
\Delta(s)=\bar Q(s,w_{best})-\bar Q(s,w_{second}).
\]

B3通过不要求所有分支确定，而要求：

1. 状态接口能够稳定保存和恢复；
2. 恢复误差和物理跳变被量化；
3. 多个状态上between-weight effect明显大于within-weight noise；
4. 最佳候选并非所有状态都相同；
5. 对不可区分候选保留并列最优或软价值标签，不强制硬分类；
6. 若接触状态恢复不可靠，后续统一改称匹配条件分支。

若不同权重效应没有超过同权重方差，则不把当前分支结果写成硬Oracle；但这不再否决
one-step-deviation价值网络路线。该路线改用独立episode分布上的期望回报标签，按6.8节
的协议评估其有效性。

## 7. B4：Oracle 预实验

B4现在采用One-step-deviation Oracle，不再要求对同一个CoppeliaSim状态进行全轨迹的
严格配对恢复。B3的状态恢复实验保留为诊断结果；B4的统计对象改为独立episode分布
上的期望单步偏离价值。

### 7.1 分支数据来源

使用以下策略采集重规划状态：

- Router；
- BestFixed；
- RGB only；
- PCD only。

必须覆盖成功、失败、恢复、遮挡、点云质量下降和专家明显分歧状态。每次
`predict_action()` 前记录部署可见观测、任务、episode seed、decision index和当前
continuation。可以保存状态快照用于调试，但不把快照恢复误差当作B4标签的主要统计量。

### 7.2 分支协议

对每个采集到的重规划状态和每个候选权重：

1. 从独立episode或独立RLBench/CoppeliaSim实例开始，避免同一实例内的分支顺序污染；
2. 在该状态用候选权重完成一次完整扩散采样；
3. 只执行一个动作chunk（当前配置为8步）；
4. chunk后统一切换到冻结的continuation，主实验使用Router；
5. 运行至固定时域或episode终止；
6. 保存chunk return、终局成功、剩余回报和部署可见观测；
7. 对RGB-only和PCD-only continuation各保留一组控制实验；
8. 用独立episode样本估计均值和置信区间，不要求不同候选在仿真器中逐状态完全一致。

分支样本建议格式：

```json
{
  "source_task": "open_drawer",
  "source_seed": 2042,
  "decision_index": 7,
  "candidate_rgb_weight": 0.75,
  "candidate_pcd_weight": 0.25,
  "diffusion_seed": 9182,
  "chunk_return": 0.0,
  "terminal_success": true,
  "continuation": "frozen_router",
  "restore_repeat": 0
}
```

### 7.3 两类首要 Oracle

Chunk Oracle：选择短时回报最高的权重，用于判断当前动作块是否有可区分性。

\[
w^K_{oracle}(s_t)=\arg\max_w G_{t:t+K}(s_t,w).
\]

One-step-deviation Oracle：候选权重只干预当前动作前缀，之后所有分支使用同一个 Router continuation，比较最终成功。

\[
w^{dev}_{oracle}(s_t)=\arg\max_w G(s_t,w;\pi_{router}).
\]

One-step-deviation Oracle 是决定是否训练长期价值模型的关键结果。

### 7.4 B4 继续条件

只有同时满足以下条件，才进入历史价值阶段一：

1. 不同状态的最佳权重有稳定差异；
2. Deviation Oracle 明显优于 BestFixed；
3. Oracle 仍明显优于当前 Router；
4. Oracle 增益在独立episode的置信区间下仍成立，而不是只出现在单一顺序或单一seed；
5. 增益不只来自单个任务或人为传感器 dropout；
6. 使用部署可见观测能够预测一部分最佳权重差异。

若 Oracle 与 Router 几乎相同，说明当前 Router 已接近可达到上界，停止复杂价值路线或缩小研究主张。若 Oracle 不高于 BestFixed，停止动态价值路线。

## 8. 进入历史价值模型后的最小顺序

B4 通过后再执行：

```text
分支数据集清洗与 train/val/test 按 episode 划分
    ↓
当前帧价值模型
    ↓
8 步 GRU 历史价值模型
    ↓
历史打乱和长度 1/4/8/16 消融
    ↓
Recurrent Router 同容量、同交互预算基线
    ↓
固定 continuation 的 Stage-1 价值训练
    ↓
Bellman Stage-2
    ↓
切换代价、优势阈值和最短保持时间
```

阶段一训练前不实现事件触发和非线性冲突算子，避免在 Oracle 尚未证明空间时扩大工程范围。

## 9. 每阶段交付物

| 阶段 | 必须交付 | 进入下一阶段条件 |
|---|---|---|
| B0 | checkpoint hash、commit、三基线结果 | 结果可复现 |
| B1 | 五档固定权重结果表 | Router 高于或可能高于 BestFixed |
| B2 | R0/R1重复性报告、80个严格配对episode | 可重复、seed对齐、日志完整 |
| B3 | V1/V2/V3 状态恢复报告 | 权重效应大于恢复噪声 |
| B4 | One-step Chunk/Deviation Oracle | 独立episode上Oracle高于BestFixed和Router |
| Stage-1 | 排序、Brier、regret、闭环结果 | 历史能预测权重效果 |
| Stage-2 | Bellman价值与同预算对照 | 高于 Recurrent Router |

## 10. 推荐的立即执行顺序

当前从这里开始：

1. 保持开发集选出的 Best Fixed 为 `[0.5,0.5]`，不再用后续结果调权重；
2. 实现 `--test-start-seed`、`--diffusion-seed` 和 `--deterministic`；
3. 按 episode 与 decision identity 生成独立 diffusion seed；
4. 实现 `episodes.jsonl` 和可选 `decisions.jsonl`；
5. 分别完成 Best Fixed 和 Router 的两次20回合重复性验收；
6. 使用新seed完成 Router 与 Best Fixed 的80回合严格配对；
7. B2通过后实现状态保存/恢复；
8. 状态恢复验收通过后采集五档候选权重的Oracle分支；
9. 只有Oracle明显超过Router和Best Fixed，才编写价值网络；
10. 价值方法完成后再做400回合确认性评测和paired bootstrap。

短期最关键的判据不是 Router 是否在当前80回合中超过固定权重，而是：

> **相同方法能否严格复现，以及同一状态下的候选权重Oracle是否仍明显超过Router和Best Fixed。**


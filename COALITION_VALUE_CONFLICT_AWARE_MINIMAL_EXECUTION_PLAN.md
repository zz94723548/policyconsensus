# 基于现有 MMPC 代码的最小执行方案

> 对应技术路线：`COALITION_VALUE_CONFLICT_AWARE_ROADMAP.md`  
> 第一阶段平台：RLBench MT4，RGB + PCD + DINO  
> 第二阶段平台：真实 UR5e，RGB + Tactile  
> 核心原则：先在现成仿真环境中完整跑通方法，并保证成功率不低于 Released MMPC，再迁移到视触任务。

## 1. 执行目标

本项目按“两阶段、一个硬门槛”执行：

```text
阶段 A：RLBench RGB + PCD + DINO
复现 Released MMPC
    ↓
模态级 Score 适配
    ↓
候选共识配置与 Oracle 分支
    ↓
历史价值模型
    ↓
周期选择、切换代价与滞回
    ↓
与 Released MMPC 同条件闭环评测
    ↓
硬门槛：总体及各子任务成功率不低于 Released MMPC
    ↓
阶段 B：RGB + Tactile 迁移与论文主实验
```

阶段 A 的目的不是证明视触结论，而是以仓库已有、可重复、可批量评测的仿真数据验证整个算法和工程闭环。只有阶段 A 达到性能非劣要求，才进入阶段 B。

## 2. 为什么先使用 RGB、PCD 和 DINO

当前代码已经提供完整的三模态仿真链路：

- 配置：`modular_policy/config/train_dp_unets_spec_rgb_pcd_dino.yaml`；
- 任务：`modular_policy/config/task/rlbench/mt4.yaml`；
- 环境：`modular_policy/env/rlbench/env.py`；
- 闭环评测：`modular_policy/env_runner/rlbench_runner.py`；
- 数据：`mt4_expert_200.zarr` 或同格式 RLBench 数据；
- 指标：整体和四个子任务的成功率。

MT4 包含 `open_box`、`toilet_seat_up`、`open_drawer` 和 `take_umbrella_out_of_umbrella_stand`。相比之下，当前 RGB+Tactile 配置对应真机拼图数据，没有现成的触觉仿真和状态克隆。因此，仿真应当是第一执行阶段，真机视触是第二阶段。

## 3. 阶段 A 的研究对象

### 3.1 从两模态权重推广到三模态共识

技术路线中的视触标量权重为：

\[
\epsilon_w=w\epsilon_V+(1-w)\epsilon_T.
\]

在三模态仿真中改为单纯形权重：

\[
\epsilon_{\boldsymbol w}=w_R\epsilon_R+w_P\epsilon_P+w_D\epsilon_D,
\qquad w_R+w_P+w_D=1,\quad w_m\ge 0.
\]

其中 \(R\) 为 RGB，\(P\) 为 PCD，\(D\) 为 DINO point cloud。长期价值模型评估候选共识配置 \(\boldsymbol w\)，迁移到视触任务时再退化为 \((w,1-w)\)。

### 3.2 保留模态内部专家结构

当前 `num_modules: 2`，三种模态共形成 6 个 UNet：

```text
RGB expert 1, RGB expert 2,
PCD expert 1, PCD expert 2,
DINO expert 1, DINO expert 2
```

不能直接把三模态权重当成 6 个专家权重。最小适配方式是：

1. 保留 Released Router 在每个模态内部的两个专家相对权重；
2. 将每个模态的两个专家合成为一个模态级 Score；
3. 只由新方法选择三个模态之间的总权重。

\[
\epsilon_m=\sum_i\alpha_{m,i}\epsilon_{m,i},
\qquad \alpha_{m,:}=\operatorname{softmax}(r_{m,:}).
\]

这样冻结原专家，只替换高层模态共识决策。

## 4. 最小候选共识集合

三模态连续单纯形搜索代价过高。第一版使用 10 个离散候选：

| 编号 | RGB | PCD | DINO | 含义 |
|---|---:|---:|---:|---|
| C0 | 1.00 | 0.00 | 0.00 | RGB only |
| C1 | 0.00 | 1.00 | 0.00 | PCD only |
| C2 | 0.00 | 0.00 | 1.00 | DINO only |
| C3 | 0.50 | 0.50 | 0.00 | RGB + PCD |
| C4 | 0.50 | 0.00 | 0.50 | RGB + DINO |
| C5 | 0.00 | 0.50 | 0.50 | PCD + DINO |
| C6 | 1/3 | 1/3 | 1/3 | 均匀三模态 |
| C7 | 0.50 | 0.25 | 0.25 | RGB 主导 |
| C8 | 0.25 | 0.50 | 0.25 | PCD 主导 |
| C9 | 0.25 | 0.25 | 0.50 | DINO 主导 |

Released MMPC 的 6 维 Router 输出作为独立基线保留，不强行映射为上述某个候选。

## 5. 成功率非劣硬门槛

阶段 A 的首要验收不是权重曲线或价值预测误差，而是闭环成功率不低于现有方法。

### 5.1 比较协议

Released MMPC 和新方法必须使用：

- 同一个专家 checkpoint、EMA 模型和 normalizer；
- 相同 `n_obs_steps=2`、`n_action_steps=8` 和 100 个扩散步；
- 相同 MT4 任务分布、环境种子和最大回合长度 250；
- 相同测试 episode 数、相机、点云和 DINO 输入；
- 不允许只对新方法挑选有利 checkpoint。

### 5.2 最小评测规模

开发检查使用 80 个 episode，即每个子任务 20 个。最终非劣检验建议使用 5 组独立种子，每组 80 个 episode，总计 400 个 episode/方法。

### 5.3 验收规则

至少同时满足：

1. 新方法整体成功率的点估计不低于 Released MMPC；
2. 四个子任务中不能有任何一个出现未经解释的明显退化；
3. 使用配对环境 seed 计算 success difference；
4. 报告 paired bootstrap 95% 置信区间；
5. 预先设置非劣界 \(\Delta\)，建议使用 2 个百分点；
6. 成功率差置信区间下界不低于 \(-\Delta\) 时才判定非劣；
7. 同时报告推理延迟、回合长度和切换次数。

若样本预算不足以完成统计非劣检验，最低工程要求是：相同 400 个配对 seed 上总体点估计不低于基线，且各任务退化不超过 2 个百分点。该结果不能直接作为论文统计结论。

## 6. 文件级最小实现

保留 Released MMPC 文件不变，新增：

```text
modular_policy/model/consensus/mmpc_expert_adapter.py
modular_policy/model/consensus/history_value.py
modular_policy/model/consensus/switch_controller.py
modular_policy/policy/value_guided_consensus.py
modular_policy/workspace/collect_consensus_branches.py
modular_policy/workspace/train_consensus_value_stage1.py
modular_policy/workspace/train_consensus_value_stage2.py
modular_policy/workspace/eval_value_consensus.py
modular_policy/config/value_consensus_rgb_pcd_dino.yaml
scripts/run_value_consensus.py
```

小范围扩展：

```text
modular_policy/env/rlbench/env.py
modular_policy/env_runner/rlbench_runner.py
```

### 6.1 `mmpc_expert_adapter.py`

- 从冻结 checkpoint 复用编码器、6 个 UNet、Router、scheduler 和 normalizer；
- 显式建立 `rgb/pointcloud/dino_pointcloud -> expert indices`；
- 保留模态内部 Router 权重；
- 返回三个模态级 Score；
- 支持传入固定三模态候选权重；
- 检查每次扩散采样期间候选配置不变。

### 6.2 `value_guided_consensus.py`

支持：

```text
released_router     原始 6 专家 softmax
fixed_candidate     固定 C0-C9
oracle_candidate    仿真分支选出的候选
value_candidate     价值模型选择候选
```

输出 `action`、`action_pred`、`selected_candidate`、`modality_weights`、`intra_modality_weights`、`candidate_values`、`previous_candidate` 和 `dwell`。

### 6.3 `history_value.py`

部署可见历史使用最近 8 个策略重规划时刻：

```text
RGB/PCD/DINO pooled features
joint_positions + gripper_joint_positions
上一执行动作摘要 + 上一候选配置
        ↓
单层 GRU
        ↓
拼接 candidate id / candidate weights
        ↓
Q_chunk + P_success
```

不输入物体真实位姿、任务成功标签或未来观测。

### 6.4 环境状态恢复

在 `RlbenchEnv` 增加：

```python
state = env.get_branch_state()
env.set_branch_state(state)
```

状态至少覆盖机器人关节与夹爪、任务物体 configuration tree、`cur_step/done`、各类 RNG、策略历史缓存和扩散 generator state。

先验收：同一状态、同一候选、同一扩散噪声重复执行的方差必须显著小于不同候选间差异。若 RLBench 接触状态不能可靠恢复，则改用相同初始 seed 的随机候选分配，不声称精确反事实 Oracle。

## 7. 阶段 A 执行顺序

### A0：复现 Released MMPC

1. 准备 MT4 数据和 RGB+PCD+DINO checkpoint；
2. 固定 checkpoint、配置与测试 seed；
3. 使用原 `DiffusionUnetsPolicy` 评测；
4. 记录总体及各子任务成功率；
5. 保存每个 episode 的成功、长度和 6 维 Router 权重。

验收：相同 seed 结果可复现，并形成不可再调参的 Released MMPC 基准表；记录 checkpoint SHA256、代码 commit 和完整 Hydra 配置。

### A1：模态适配与固定候选

1. 实现三模态 Score 适配；
2. 逐个运行 C0-C9；
3. 验证一次 `predict_action()` 内候选配置固定；
4. 使用相同 seed 测试固定候选成功率；
5. 只在验证集选择最佳固定候选。

验收：Released 模式与原代码输出一致；所有候选动作有限且 shape 正确；专家和编码器冻结；得到单模态、两两组合、均匀组合和最佳固定组合基线。

### A2：配对分支与 Oracle

1. 在每个重规划点保存可恢复状态；
2. 对 C0-C9 使用相同状态、历史和扩散初始噪声展开；
3. 执行相同长度动作前缀；
4. 之后统一切回 Released MMPC continuation；
5. 保存 Chunk return、最终成功和下一历史状态；
6. 构建 Chunk Oracle 和 One-step-deviation Oracle。

只有同时满足以下条件才进入 A3：最佳候选随状态稳定变化；Oracle 优于最佳固定候选；候选效应大于恢复噪声；Oracle 相对 Released MMPC 存在可学习空间；提升不是人为删除模态造成的。

### A3：阶段一历史价值模型

训练目标：

\[
\mathcal L=\operatorname{Huber}(\hat Q_K,G_K)
+\lambda_s\operatorname{BCE}(\hat P_{succ},y_{succ})
+\lambda_r\mathcal L_{rank}.
\]

数据按 episode、任务和 seed 分组切分。必做对照为当前帧价值、8 步历史价值、历史打乱、Recurrent Router、Released MMPC、最佳固定候选和 Oracle。离线指标为 Brier score、候选排序相关、top-1 准确率和相对 Oracle regret。

### A4：阶段二 Bellman 价值

收集 `(history, candidate, chunk_return, next_history, done)`，使用离散候选最大值和 target network：

\[
y=R+\gamma^\tau(1-done)\max_{c'\in\mathcal C}\bar Q(h',c').
\]

使用阶段一初始化；不做连续权重外推；Recurrent Router 获得相同仿真交互预算；超参数只在独立验证集选择。

### A5：切换控制与最终闭环

\[
U_t(c)=\hat Q(h_t,c)-\lambda_{sw}
\lVert\boldsymbol w_c-\boldsymbol w_{prev}\rVert_2^2.
\]

依次增加直接最大价值、切换代价、优势阈值和最短保持时间。本阶段结束后按第 5 节完成非劣评测；未通过成功率门槛，不进入真机迁移。

## 8. 阶段 A 最小配置

```yaml
consensus:
  mode: released_router
  candidates:
    - [1.00, 0.00, 0.00]
    - [0.00, 1.00, 0.00]
    - [0.00, 0.00, 1.00]
    - [0.50, 0.50, 0.00]
    - [0.50, 0.00, 0.50]
    - [0.00, 0.50, 0.50]
    - [0.333333, 0.333333, 0.333333]
    - [0.50, 0.25, 0.25]
    - [0.25, 0.50, 0.25]
    - [0.25, 0.25, 0.50]
  preserve_intra_modality_router: true

history: {length: 8, hidden_dim: 256, num_layers: 1}
switch: {lambda_sw: 0.0, advantage_threshold: 0.0, min_dwell: 1}
evaluation:
  paired_seeds: true
  episodes_per_run: 80
  num_seed_groups: 5
  noninferiority_margin: 0.02
```

## 9. 阶段 A 结果表

| 方法 | Overall | Open Box | Toilet Up | Open Drawer | Umbrella | 延迟 | 切换/回合 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Released MMPC |  |  |  |  |  |  |  |
| Best fixed candidate |  |  |  |  |  |  | 0 |
| Recurrent Router |  |  |  |  |  |  |  |
| Stage-1 value |  |  |  |  |  |  |  |
| Stage-2 value |  |  |  |  |  |  |  |
| Value + switch control |  |  |  |  |  |  |  |
| Oracle |  |  |  |  |  | — |  |

非劣报告还必须包含“新方法 - Released MMPC”、paired bootstrap 95% CI、非劣界、各子任务退化和是否允许进入阶段 B。

## 10. 阶段 B：迁移到 RGB + Tactile

只有阶段 A 通过非劣门槛后执行。历史价值定义、两阶段训练、切换控制、数据日志和评测代码保持不变。只做以下替换：

- 三模态适配器改为 RGB/Tactile 两模态；
- 候选集合改为 \(\{0,0.25,0.5,0.75,1\}\)；
- RLBench 状态和回报接口改为真机匹配初始条件或未来视触仿真接口；
- 历史输入改为视觉、触觉、机器人状态、动作和上一权重。

阶段 A 只能证明算法适用于 MMPC 多模态共识，最终“视触长期价值与冲突感知”的论文主张仍必须由阶段 B 验证。

## 11. 暂不进入主流程的内容

- 事件触发重规划；
- 非线性视觉主导/触觉主导算子；
- 连续单纯形权重优化；
- 单独的不确定性网络；
- 新增模态能力验证；
- 真机在线强化学习。

只有价值选择达到 Released MMPC 成功率，且 Oracle 显示单纯形共识仍有稳定不足时，才开发冲突感知非线性算子。

## 12. 最终执行优先级

```text
P0  复现 RGB+PCD+DINO Released MMPC
P1  三模态 Score 适配与固定候选
P2  RLBench 状态恢复验收
P3  Chunk/Deviation Oracle
P4  8 步历史阶段一价值模型
P5  Bellman 阶段二
P6  切换代价、滞回与最短保持时间
P7  400 个配对 episode 的非劣评测
P8  通过后迁移到 RGB+Tactile
P9  必要时增加非线性冲突共识
```

最早可运行闭环是 P0-P1；最早可验证研究假设的是 P0-P4；允许进入真实视触实验的必要条件是 P0-P7 全部完成并通过成功率非劣门槛。

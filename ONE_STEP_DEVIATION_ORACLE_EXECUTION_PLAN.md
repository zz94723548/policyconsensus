# RGB + PCD One-step-deviation Oracle 执行计划

## 1. 目标与定义

在重规划状态 (s_t)，候选权重 (w=(w_{rgb},1-w_{rgb})) 只影响当前一个动作
chunk（当前为8步），随后切换到固定 continuation。定义：

\[
Q^{\pi_c}_{1-dev}(s_t,w)=E[G(s_t; w\text{ for one chunk},\pi_c\text{ afterwards})].
\]

主 continuation 为冻结 Router，RGB-only 和 PCD-only 为控制组。这里“一步”是一个
动作 chunk，不是 diffusion denoising timestep，也不是完整 episode。该路线受到
`Compose Your Policies!` 的单步凸组合误差理论启发，但仍需独立 episode 实验验证。

## 2. 不采用旧 B3 分支作为 B4 主实验的原因

B3 已显示同一 CoppeliaSim 实例连续恢复存在执行顺序污染。因此 B4 使用独立进程采集，
统计对象是独立 episode 分布上的期望 one-step deviation value。相同任务与 seed 是
统计分组标识，不保证跨进程观测逐位相同；每个分支必须保存自己的决策前观测和历史。
观测哈希不一致作为诊断字段记录，不再中止已经完成的整批采集。

## 3. 总体流程

```text
采集多个决策阶段的部署可见历史
  -> 五档权重分别执行一个动作chunk
  -> 固定continuation运行到终止
  -> 保存终局成功、回报、chunk进度和风险标签
  -> 聚合条件成功率/辅助指标及置信区间
  -> 训练历史条件多头 V(H,w)
  -> 在线每个chunk选择一次权重
  -> 低置信度时回退Router
  -> 与Router、BestFixed、单专家比较
```

## 4. B4.1 采集接口

新增：

```text
scripts/collect_one_step_oracle.py
modular_policy/workspace/collect_one_step_oracle.py
```

CLI 必须支持：

```text
--checkpoint --task --episodes --test-start-seed --diffusion-seed
--decision-index --weights --prefix-actions --continuation
--num-repeats --independent-process --output --device
```

每个 episode 由 Router 运行到 `decision-index`，保存从 episode 开始至当前决策点的
部署可见观测历史。每个 `(episode seed, decision index, weight, repeat)` 都在独立
RLBench/CoppeliaSim 过程执行，避免同一模拟器隐藏状态累积。repeat 使用不同环境 seed；
同一 repeat 下的五档权重共享环境 seed 和 diffusion seed，以形成 seed-block 比较。

## 5. B4.2 分支协议

1. 独立初始化任务和 episode/repeat seed；
2. 运行到目标 decision index；
3. 用候选权重完成一次完整 diffusion action prediction；
4. 执行 `prefix_actions=8` 个动作；
5. 切换到固定 continuation；
6. 运行至成功、失败或最大时域；
7. 记录 chunk return、终局成功、折扣回报、任务进度变化/风险标签和 continuation 长度；
8. 关闭仿真器。

若 episode 在目标决策点前已经终止，记录 `branch_valid=false` 和原因并跳过该分支，不
让不可达的晚期决策点中止整批采集；成功率聚合只纳入有效分支。

不同独立实例不要求物理轨迹逐位相同，标签估计的是独立 episode 分布上的期望表现。
相同 seed 的五个权重分支可能出现观测差异；保留各自的真实分支特征，记录
`canonical_observation_match` 和差异哈希。差异不导致静默覆盖或整批失败，统计比较按
episode seed 分组并用分组 bootstrap 估计不确定性。

## 6. 日志格式

`states.jsonl`：

```json
{"state_id":"open_drawer_6202_d4_r0","task":"open_drawer","environment_seed":6202,"decision_index":4,"source_policy":"router","state_feature_path":"features/open_drawer_6202_d4_r0_w0.75.pt","canonical_observation_match":true,"branch_features":[{"weight_rgb":0.75,"path":"features/open_drawer_6202_d4_r0_w0.75.pt","sha256":"..."}]}
```

`branches.jsonl`：

```json
{"state_id":"open_drawer_6202_d4_r0","candidate_rgb_weight":0.75,"candidate_pcd_weight":0.25,"continuation":"router","prefix_actions":8,"chunk_return":0.0,"episode_return":1.0,"discount_factor":1.0,"discounted_return":1.0,"terminal_success":true,"repeat":0,"environment_seed":6202,"branch_valid":true,"independent_process":true,"state_feature_path":"features/open_drawer_6202_d4_r0_w0.75.pt","state_feature_sha256":"..."}
```

`episode_return` 是从决策点开始到终止的未折扣回报；`discounted_return` 使用显式记录的
`discount_factor` 计算。状态文件保存 `current` 观测和从 episode 起点到当前决策点的
`history`。训练时每条有效 branch 使用自己的 `state_feature_path`；同 seed 分支只作为
统计 block，不假定其特征完全相同。

`summary.json` 必须保存 checkpoint、git commit、seed、权重、continuation、设备、
分支数量和运行参数。

## 7. B4.3 运行阶段

### Smoke：1个状态、2个权重、1次重复

```bash
PARALLEL_RLBENCH=0 python scripts/collect_one_step_oracle.py \
  --checkpoint output/router_latest.ckpt --task mt4 --episodes 1 \
  --test-start-seed 6202 --diffusion-seed 65000 --decision-index 4 \
  --weights 0,1 --prefix-actions 8 --continuation router \
  --num-repeats 1 --independent-process \
  --output output/rgb_pcd_router_study/oracle/smoke --device cuda:0
```

必须检查两个候选分支、权重值、Router continuation、chunk长度和三个日志文件。

### Pilot：20个状态、5档权重、2次独立重复

```bash
PARALLEL_RLBENCH=0 python scripts/collect_one_step_oracle.py \
  --checkpoint output/router_latest.ckpt --task mt4 --episodes 20 \
  --test-start-seed 6500 --diffusion-seed 66000 --decision-index 4 \
  --weights 0,0.25,0.5,0.75,1 --prefix-actions 8 \
  --continuation router --num-repeats 2 --independent-process \
  --output output/rgb_pcd_router_study/oracle/pilot_router --device cuda:0
```

分支数为 `20×5×2=200`。使用完全相同状态和参数，再分别执行 `--continuation rgb`
和 `--continuation pcd`，输出目录使用 `pilot_rgb` 和 `pilot_pcd`。

### Stage scan：先检查不同决策阶段是否存在权重偏好

当前 pilot 只测了 `decision_index=4`，不能据此判断其它阶段。先对四个 MT4 子任务各取
两个 seed，在早、中、较晚决策点做小规模诊断；这一轮用于发现阶段条件信号，不用于最终
性能主张或训练集定稿。决策点前已结束的 episode 会记为无效分支并跳过。

```bash
for decision in 2 6 10; do
  PARALLEL_RLBENCH=0 python scripts/collect_one_step_oracle.py \
    --checkpoint output/router_latest.ckpt --task mt4 --episodes 8 \
    --test-start-seed 6700 --diffusion-seed 68000 \
    --decision-index "$decision" --weights 0,0.25,0.5,0.75,1 \
    --prefix-actions 8 --continuation router --num-repeats 1 \
    --independent-process \
    --output "output/rgb_pcd_router_study/oracle/stage_d${decision}" \
    --device cuda:0
done
```

每个阶段目录预期最多 `8×5=40` 个分支。当前采集器可直接用于检查有效分支数、四任务覆盖、
每个分支历史特征、`canonical_observation_match`、成功率和回报。chunk 进度与风险标签需先
依据各任务可用的 simulator predicate 定义并实现，再用于辅助头训练。若某阶段出现
可重复的条件偏好，再增加该阶段样本量和独立 repeats；不要用这一小轮估计最终置信区间。

## 8. B4.4 标签聚合

对每个 `(task, decision_index, state_id, weight, continuation)` 计算：

```text
success_rate, success_ci, progress_mean/std, return_mean/std, risk_rate, num_independent_repeats
```

保留每条分支的原始标签。成功率置信区间以独立 repeat 为采样单位，并按 episode seed
分组 bootstrap；同一 seed 下五个权重不可拆开到不同数据切分。平均权重成功率相同不代表
条件偏好不存在，因此还要报告任务、决策阶段和历史条件分层结果。若各条件下权重的成功、
进度和风险均无可重复差异，就没有可学的选权信号。当前采集器尚未生成通用的进度/风险
标签；须先逐任务定义并实现后，才能进入对应辅助头训练。

成功率是主预测目标；进度与风险是辅助头，不把它们事后混成任意加权总分。输出
`success_labels.json`，保留每次 rollout 的 Bernoulli 标签供 BCE 训练，并输出分组成功率
及区间作诊断。第一版不强制构造单一“最佳权重”标签；只有样本量足够且候选成功率区间
稳定分离时，才额外生成 pairwise preference 标签。

## 9. B4.5 价值网络

第一版输入决策前 current RGB/PCD encoder feature、最近可用观测历史和候选权重；先用
紧凑历史编码，只有数据量与 held-out 结果支持时才增加 GRU。新增：

```text
modular_policy/model/value/one_step_value_net.py
modular_policy/workspace/train_one_step_value.py
```

推荐接口：

```python
predictions = value_net(history_features, candidate_weights)
# outputs: success_logit, expected_progress, risk_logit
```

成功头用 BCE，进度头用 Huber，风险头用 BCE；各辅助损失及归一化须在查看 test 结果前
固定。按 episode seed 分组划分 train/val/test。若数据仍不能区分权重，先扩充有差异的
阶段/任务数据，不通过人为放大无关指标制造 label 差异。

## 10. B4.6 在线部署

```python
candidate_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
predictions = value_net(history_features, candidate_weights)
success_prob = predictions['success_logit'].sigmoid()
w_rgb = candidate_weights[success_prob.argmax()]
w_pcd = 1.0 - w_rgb
action_chunk = policy.predict_action(fixed_weights=(w_rgb, w_pcd))
execute(action_chunk[:8])
```

价值网络每个重规划 chunk 调用一次，不参与扩散内部每个 denoising timestep。用 ensemble
分歧或预测区间评估置信度：若候选成功概率无法可靠区分、最佳候选没有超过 Router 的
预设优势阈值，或输入超出训练支持范围，则直接回退到 Router。纯视觉只有在独立验证集
上证明表现不劣于 Router 后才可作为回退。报告 fallback 次数及其成功率。回退是部署保护，
不是价值网络有效性的证据。第一版不实现事件触发或 Bellman Stage-2。

## 11. B4.7 最终对照

比较：RGB-only、PCD-only、BestFixed、原始 Router、One-step value+Router continuation、
One-step value+RGB continuation、One-step value+PCD continuation。所有方法使用相同
MT4 测试任务和 episode seed，报告 overall/per-task success、95% grouped bootstrap CI、
预先定义的进度/风险指标、平均权重、权重熵、推理开销和 fallback 次数。

## 12. 通过条件和失败处理

进入历史价值网络前必须满足：

1. One-step Oracle 在独立 episode 上高于 BestFixed；
2. One-step Oracle 高于 Router，或统计不劣但显著降低部署开销；
3. 增益不是只来自一个任务或一组 seed；
4. 价值网络在 held-out episode 和决策阶段上仍能预测成功概率/权重偏好；
5. 辅助进度/风险指标带来的选权提升也必须在终局成功或预先声明的次级指标上验证；
6. fallback 频率与 fallback 成功率单独报告，不能计作网络选权成功。

若 Oracle 不高于 BestFixed，先检查 decision index、prefix length、候选权重和 continuation；
若 Oracle 高于 BestFixed 但不高于 Router，说明 Router 已接近 one-step 上界；若只在单个
任务有效，则缩小论文主张，不宣称通用动态路由。

## 13. 当前执行顺序

1. Smoke 已跑通；确认固定权重、chunk、Router continuation 和特征落盘；
2. 用不同 repeat 环境 seed 修正采集，并保存每个决策点的完整部署可见历史；
3. 对 `decision_index=2,6,10` 运行 Stage scan，标记提前结束的无效分支；
4. 按 task/stage 检查成功率、回报、有效样本数和分支观测哈希差异；
5. 对出现可重复权重偏好的阶段扩大独立样本与 repeats；
6. 定义并实现 task-specific 进度/风险标签后，训练成功率主头 + 辅助头的历史条件价值网络；
7. 加入不确定性回退 Router，并在 held-out seeds/stages 验证；
8. 再运行最终方法与 Router、BestFixed、单专家对照；
9. 若自然 RGB/PCD 条件仍没有可学习的偏好，报告该组合的适用边界，再评估更具差异的专家组合。


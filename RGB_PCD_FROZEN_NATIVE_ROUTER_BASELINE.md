# 冻结 RGB/PCD 专家的原生 MMPC 路由对照

状态：已实现并通过57项相关合成检查，未启动真实训练或仿真。基于项目已有路由结构、soft-gating规则、训练/验证循环和 `scripts/eval_policy.py sim`，不另写仿真评测器。

## 1. 对照目的与冻结范围

固定现有两个专家：

```text
output/experts/rgb_seed42/checkpoints/latest.ckpt
output/experts/pcd_seed42/checkpoints/latest.ckpt
```

使用其中原有EMA参数，包括感知编码器、去噪网络、机器人状态/动作归一化。仅训练独立路由网络。专家在训练中根本不进入优化器；训练直接读取它们已经冻结产生的特征和配对误差缓存。训练前、训练后和仿真加载时校验原专家checkpoint哈希，不能在过程中悄悄换模型。

本基线直接学习权重，不预测二阶矩，也没有成功率价值网络。与路线A比较时，主要区别是“直接优化权重”和“回归二阶矩后解析计算权重”。

## 2. 复用项目里的什么

`modular_policy/policy/dp_unets_spec.py` 原 `DiffusionUnetsPolicy` 的路由结构为：

```text
联合专家特征 → Linear(D,D) → ReLU → Linear(D,专家数) → softmax
```

将原实现提取成 `make_weight_predictor`、`soft_gating_weights`，原策略也调用这两个函数，其默认结构和softmax计算不变。独立基线直接复用这些函数，而不新发明一个路由网络。

当前两专家各一个，总计2个输出。原某些组合配置的 `num_modules=2` 会每模态构建两个网络，不能直接拿来代表现在这两个既有专家。

路由特征按原联合编码器语义排列：RGB特征、PCD特征、机器人状态一次。RGB条件402维、PCD条件146维，两份共享机器人状态均为18维，因此联合输入是 `402+146-18=530`，不能将状态重复拼接成548维。实际网络为 **530→530→2，共282,492个参数**，输出顺序RGB、PCD。

不增加GRU，不额外对专家输出特征拟合标准化，保持原生路由器当前规划观测输入。当前规划观测本身仍包括专家使用的2个观测时刻。`history_contexts=1` 表示不额外叠加过去规划特征，与A/B主模型的4次规划历史存在差异；后续如果做容量/历史严格消融，应另列实验。

## 3. 损失与缓存等价性

原soft-gating路由器最小化：

$$L=E\|w(s)\hat\epsilon_{rgb}+[1-w(s)]\hat\epsilon_{pcd}-\epsilon\|^2/D.$$

冻结专家后，对同一个有限噪声样本池，该目标精确等于：

$$L=w(s)^2A(s)+[1-w(s)]^2B(s)+2w(s)[1-w(s)]C(s).$$

因此可以使用既有 `moments` 缓存直接计算梯度，不重新运行大专家。这里A/B/C是已采集误差的固定充分统计量，不是另一个预测网络的输出，也不是让路由器回归A/B/C。模型只有2个logit输出，经softmax得到权重；没有“正确专家”分类标签或成功率标签。

这个等价是对**相同冻结特征、噪声样本池和均匀扩散步权重**而言。原始在线训练循环会每次重新采样噪声，而此对照重用有限缓存，不声称完整复现原论文全部训练流程。缓存训练/验证各使用每步4/8次噪声重复，覆盖100个扩散步；各完整episode均衡。

推理每次规划计算一次路由概率，同一规划内100个去噪步共用。融合仍在噪声预测层、同一带噪动作上进行，沿用原DDPM采样循环，不对两个最终动作简单平均。这是与路线A相对应的直接路由基线，不是路线B的逐扩散步路由。

## 4. 公平数据和预算

复用 `output/paired_errors/latest_seed2026`：同样20条训练轨迹、8条验证轨迹，339/142个规划上下文，源划分为 `splits/mt4_risk_model_seed142.json`。

不使用专家训练集再增加路由训练数据，不读取16条校准或16条最终测试标签。虽然缓存metadata含全量划分ID，训练实际只读取20/8分区的特征/标签。

预算与当前统计模型对照一致：500epochs、seed42、batch64、AdamW lr3e-4、weight_decay1e-4、clip1、不早停、每25epochs保存阶段快照。原项目其他训练配置可能使用1e-4和cosine调度；本对照为匹配本轮预算采用现有统计训练循环的设置，而非宣称沿用所有原始默认超参数。

`best.ckpt` 与 `best_risk.ckpt` 均按验证集融合epsilon MSE最低选择；没有矩回归指标。`latest.ckpt` 保存末轮，并保留25~500轮的20个快照。模型参数量比A/B更大，应在结果中一并报告，不能将性能变化完全归因于学习目标。

## 5. 后台训练命令

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus

nohup python -u scripts/manip_policy.py --config-name=train_frozen_router \
  > output/train_frozen_router.log 2>&1 < /dev/null &
echo $! > output/train_frozen_router.pid

tail -f output/train_frozen_router.log
```

输出到独立目录 `output/routers/native_mmpc_seed42/`，不会覆盖专家或路线A/B模型。需要恢复时加 `resume=true`，要求数据、配置和路由源码签名相同。修改原训练循环的源码会使旧运行的恢复签名变化，已完成A/B模型的推理不受影响。

关键文件：

| 文件 | 含义 |
| --- | --- |
| `checkpoints/best_risk.ckpt` | 默认待评测路由模型 |
| `best_risk_validation.json` | RGB/PCD/等权/训练全局固定/固定0.6/直接路由的验证风险 |
| `expert_freeze_audit.json` | 训练前后专家文件未变化、优化器只含路由参数的检查 |
| `training_audit.json` | 数据划分、530维联合输入、网络参数量、实现来源与哈希 |
| `logs.json.txt` | 每轮train_router_loss、val_router_loss、融合风险 |
| `run_summary.json` | 500轮预算完成情况与最佳验证风险 |
| `cache_metadata.json` | 原配对误差缓存与专家来源 |

checkpoint仅保存小路由器和优化器，不复制两个大专家；部署仍需保留原专家文件。日志中router_loss是融合epsilon MSE，不是成功率，也不是二阶矩回归损失。

## 6. 原入口前台仿真

等训练完成并确认freeze audit后执行：

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus
export PARALLEL_RLBENCH=1
export QT_QPA_PLATFORM=xcb
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

xvfb-run -a python -u scripts/eval_policy.py sim \
  -c output/experts/rgb_seed42/checkpoints/latest.ckpt \
  --pcd-checkpoint output/experts/pcd_seed42/checkpoints/latest.ckpt \
  --fusion router \
  --router-checkpoint output/routers/native_mmpc_seed42/checkpoints/best_risk.ckpt \
  --eval-seed 42 --test-start-seed 1000 \
  -o output/eval_native_router_seed42 -n 80 -d cuda:0
```

8环境并行、四任务各20次，共80次；沿用每组8条录像及初始观测审计。无需也不允许同时传 `--calibration` 或 `--statistics-checkpoint`。路由基线没有鲁棒半径。

`eval_log.json`保存成功率；`fusion_config.json`记录专家/路由来源；`fusion_trace.json`保留实际RGB权重，moments为空，因为路由不在线预测矩阵。原runner里的avg_raw_weight字段在此表示已softmax的实际概率，不是原MMPC原始logit。

现有1000~1079种子已经用于开发/搜索，只作同协议探索对照，不称为全新独立测试。形成最终比较时，冻结所有方法并在同一组新环境种子上一起测试；相同seed也不能保证物理初始状态逐位相同。

## 7. 检查范围

57项相关合成测试通过，新增验证原路由工厂与原两层MLP结构一致、共享状态只拼一次、缓存风险及梯度与显式融合噪声MSE相同、小型合成缓存训练/保存/恢复、专家文件保持、优化器仅含路由参数、模型加载和原采样循环使用实际路由概率。没有执行真实训练、真实专家推理、仿真或硬件操作。

```bash
python -m pytest tests/test_frozen_router.py tests/test_error_moments.py \
  tests/test_error_moment_fusion.py tests/test_moment_calibration.py \
  tests/test_fixed_weight_search.py -q
```

## 8. 本次训练完成后的统一目录测试

已核对用户完成的训练：500epochs，专家冻结审计通过。`best_risk.ckpt` 对应epoch227（完成第228轮），验证融合epsilon MSE为0.002321563093805022。本节准备两个测试入口和比较汇总，尚未启动真实测试。

### 8.1 仿真成功率

使用第6节相同的环境设置，但输出目录按本次要求改为 `output/eval_matched5_seed42/router`：

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus
export PARALLEL_RLBENCH=1
export QT_QPA_PLATFORM=xcb
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

xvfb-run -a python -u scripts/eval_policy.py sim \
  -c output/experts/rgb_seed42/checkpoints/latest.ckpt \
  --pcd-checkpoint output/experts/pcd_seed42/checkpoints/latest.ckpt \
  --fusion router \
  --router-checkpoint output/routers/native_mmpc_seed42/checkpoints/best_risk.ckpt \
  --eval-seed 42 --test-start-seed 1000 \
  -o output/eval_matched5_seed42/router -n 80 -d cuda:0
```

前台运行，8环境并行、每任务20次。保留旧五组目录，结果进入独立router子目录。

### 8.2 相同16条示范上的离线误差

复用原离线评测workspace增加路由分支，不重新采集、不重新训练、不修改校准半径。只加载小路由器做特征前向，不加载大专家或A统计模型做新前向。直接读取原 `test_latest_seed2026` 的缓存，以及A最终报告对应的已保存条件/鲁棒权重，重新统一计算七种方法的风险。

```bash
python -u scripts/manip_policy.py --config-name=evaluate_frozen_router_test
```

输出位于：

```text
output/eval_matched5_seed42/offline_errors/router/
  offline_error_report.json
  offline_error_predictions.npz
  offline_error_comparison.csv
  offline_error_config.yaml
```

报告比较RGB、PCD、等权、固定0.6、A条件权重、A鲁棒权重和直接路由；包括总体、逐任务、逐episode及逐扩散步的epsilon MSE。缓存与原最终报告的signature、完整episode IDs、anchor和目标矩阵必须相同；重新计算的六种旧方法总体风险必须与原报告数值一致。原最终报告/预测不覆盖，重复输出拒绝静默覆盖。

这里是已经查看过的16条最终示范上的追加路由对照，未用它选路由checkpoint。它不等于新的未查看测试集，也不能与闭环成功率混为一个指标。

### 8.3 合并已有结果

两项测试完成后执行：

```bash
python scripts/eval_policy.py compare-matched -i output/eval_matched5_seed42
```

该命令只读结果，不运行模型，生成根目录下 `comparison.csv` 和 `comparison.json`，显示原五组与router的成功率和离线风险；如果存在既有固定0.6和B仿真日志，也会从其原目录读取并列出，保留来源路径。B目前没有相同最终离线结果时该栏留空，不编造数值。

汇总检查专家哈希、环境/扩散seed、任务分配和测试次数；路由仿真与离线报告还必须对应同一router checkpoint哈希。物理初始状态仍可能存在先前已记录的细微差异，因此这些历史批次不是逐位完全相同物理状态的对照。

新增完整合成离线评测与结果对比检查后，相关测试共58项通过；没有运行真实仿真或真实模型前向。增加的是推理加载/报告接口，原生路由结构和已训练权重未改变。

## 9. 按用户指定改用专家训练数据的路由对照

用户指出原先路由使用训练专家的数据。本节新增该协议，保留前述20/8对照，不用它覆盖或冒充专家数据路由结果。

新配置使用原 `expert_train` 120条（每任务30条）训练路由、`expert_val` 20条（每任务5条）验证；不使用原 `risk_train` 28条、校准16条或最终测试16条训练/选模。两位专家依旧冻结。网络、学习率、500轮预算、不早停及保存规则与先前路由对照相同；训练轨迹增多，固定轮数下优化更新总数也会增加。

需要先为这140条原有轨迹计算冻结专家特征和有限噪声样本池的风险缓存。此步骤不是采集新演示，不重训专家。仍按现有规划间隔8选上下文、100个扩散步、训练/验证4/8次噪声重复计算；它与原项目逐帧重新采样噪声的完整在线训练流程仍有区别。本节改变的是用户指定的数据来源，不声称复现所有旧训练细节。

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus

nohup bash -c '
set -e
python -u scripts/manip_policy.py --config-name=collect_frozen_router_expert_data
python -u scripts/manip_policy.py --config-name=train_frozen_router_expert_data
' > output/train_frozen_router_expert_data.log 2>&1 < /dev/null &
echo $! > output/train_frozen_router_expert_data.pid

tail -f output/train_frozen_router_expert_data.log
```

第一步成功完成后才执行第二步；失败时set-e停止，不误用不完整缓存。新缓存 `output/paired_errors/router_expert_data_seed2026` 保存特征/二阶矩，不保留巨量完整残差；新模型输出 `output/routers/native_mmpc_expert_data_seed42`，专家与旧20/8路由均保留。

采集步骤中断后，可单独执行 `python -u scripts/manip_policy.py --config-name=collect_frozen_router_expert_data resume=true` 恢复；随后执行新训练配置。训练恢复则对第二条训练命令加 `resume=true`。不要将旧20/8缓存仅改名后用于本配置，加载器会检查120/20覆盖和角色隔离。

59项相关合成检查通过，新增加120/20分区与保留集隔离、训练数据类/路由构建检查；没有启动真实缓存构造或训练。此前20/8来源的路由结果应标为额外数据预算对照；新模型以后对比时需标明专家训练数据复用，不能混合两类训练来源。

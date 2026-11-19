# 路线 B：一次预测逐扩散步二阶矩与融合权重

状态：代码与合成检查已完成，尚未启动真实训练或仿真。路线 A 的实验报告、checkpoint、缓存、校准半径及最终报告保留，不覆盖。

## 1. 与路线 A 的区别

这里的“阶段”明确指一次重新规划：当前配置每次预测16步动作，执行其中8步，再重新规划；每次规划内部执行100个扩散去噪步。

| 项目 | 路线 A | 路线 B |
| --- | --- | --- |
| 每次重规划的统计前向次数 | 1 | 1 |
| 统计网络输出 | 3个数 | 3K个数；本次K=100，即300个数 |
| 输出形式 | A(s)、B(s)、C(s) | 各步A_k(s)、B_k(s)、C_k(s) |
| 同次规划内的权重 | 一个w(s)，所有去噪步共用 | 每个噪声时间步使用对应w_k(s) |
| 真实执行中是否知道误差真值 | 不需要 | 不需要 |

网络实际输出张量为 `[batch, K, 3]`，不是重复同一个矩阵K次。顺序严格按原缓存的 `timesteps`：当前为 `[99,98,...,0]`。输出第0组对应噪声步99，第99组对应噪声步0，不能将输出索引直接当作噪声时间步编号。

## 2. 标签、模型和目标

复用 `output/paired_errors/latest_seed2026` 中的 `moments_by_step`。每个规划上下文已有100组二阶矩，不需要重新运行专家采集。专家仍为RGB/PCD的latest EMA，动作归一化、观测特征、原始数据划分与噪声协议均不变。

每步标签是：

$$A_k=E[\|\hat\epsilon_{r,k}-\epsilon\|^2/D],\quad
B_k=E[\|\hat\epsilon_{p,k}-\epsilon\|^2/D],\quad
C_k=E[\langle\hat\epsilon_{r,k}-\epsilon,\hat\epsilon_{p,k}-\epsilon\rangle/D].$$

同一个标签内两专家读取同一带噪动作、同一噪声等级和同一噪声真值。各k的误差已经保存在缓存；这里直接以逐步标签训练，不先平均成路线A的3个目标。

仍采用RGB 402维、PCD 146维特征，各投影64维，GRU hidden64、head中间层32。最后Linear从输出3维改为300维，各组三元组独立通过softplus/tanh结构形成PSD矩阵。参数量由74,883增加到84,684，其余主干宽度不变。

每个噪声步的尺度仅在20条统计训练轨迹上拟合：

$$s_k=\max\left(\frac{\bar A_{k,train}+\bar B_{k,train}}2,10^{-8}\right).$$

均值先在各episode内部计算，再对episode均衡平均。模型输出及矩回归损失使用固定s_k；不能在验证、校准、最终测试重新拟合。原A的单一输出尺度保持原实现。

逐上下文训练损失为：

$$L(s)=\sum_k\lambda_k\left[
\left(\frac{\hat A_k-A_k}{s_k}\right)^2+
\left(\frac{\hat B_k-B_k}{s_k}\right)^2+
2\left(\frac{\hat C_k-C_k}{s_k}\right)^2\right],\quad\lambda_k=1/K.$$

逐步归一化是本次B新增的优化设置，用于减少不同噪声等级误差量级差异对学习的影响。A/B的归一化矩损失数值因此不能直接比较；两者未经归一化的融合epsilon风险和闭环成功率仍可作为共同指标。

使用与A相同的20条训练、8条验证，339/142个上下文，历史4次规划、batch64、AdamW lr3e-4、weight_decay1e-4、梯度裁剪1、seed42、500epochs、不早停。每25epochs保留阶段快照；每轮保存latest；best按验证矩损失，best_risk按验证融合epsilon风险。默认仿真使用best_risk，继续与路线A的选模标准对齐。

历史1消融可通过 `history_contexts=1` 和独立输出目录执行，但不是本轮默认实验。没有修改或重训专家。

## 3. 逐步解析融合

统计模型每次规划前向一次得到整组矩阵，分别解析：

$$w_k(s)=\operatorname{clip}\left(
\frac{\hat B_k(s)-\hat C_k(s)}
{\hat A_k(s)+\hat B_k(s)-2\hat C_k(s)},0,1\right).$$

每个去噪步读取对应权重：

$$\hat\epsilon_{mix,k}=w_k(s)\hat\epsilon_{rgb,k}+(1-w_k(s))\hat\epsilon_{pcd,k}.$$

仍使用一个共享带噪动作和原有 `DiffusionUnetPolicy.conditional_sample` 循环；两个专家计算当前噪声预测，然后融合并只执行一次scheduler更新。不会对K个权重求平均后再部署。分母退化时使用训练组拟合的全局回退权重，容差沿用1e-8。

离线风险也先逐步计算，再按λ汇总：

$$R_B(s)=\sum_k\lambda_k\,[w_k^2A_k+(1-w_k)^2B_k+2w_k(1-w_k)C_k].$$

不能把Gamma先跨步平均、或把w_k先平均，再代入路线A风险公式；非线性平方项使它们一般不相等。训练/选模/报告均已支持实际的逐步风险。

这个目标没有建模不同扩散步误差之间的完整跨步传播或交叉矩阵，不能由此直接保证整条生成动作或任务成功率最优。

## 4. 第一阶段先验证条件逐步权重

本轮先实现B的条件点估计权重。路线A的标量半径是针对跨步汇总矩校准的，不能直接复用到B的100组矩阵。

因此B仿真使用 `--fusion point --statistics-checkpoint ...`，不传 `--calibration`。同时传两种来源或要求用此入口执行robust会明确报错。B的逐步鲁棒集合与校准方案尚未实现，后续若研究应独立设计并报告。

## 5. 训练命令：用户自行启动后台训练

```bash
cd /home/dbdx/NEUZZ/policyconsensus
conda activate policy-consensus

nohup python -u scripts/manip_policy.py --config-name=train_error_moments_per_step \
  > output/train_error_moments_per_step.log 2>&1 < /dev/null &
echo $! > output/train_error_moments_per_step.pid

tail -f output/train_error_moments_per_step.log
```

Ctrl+C停止tail，不会停止后台训练。训练输出隔离在：

```text
output/error_moments/per_step_history4_seed42/
  training_audit.json
  cache_metadata.json
  baselines.json
  risk_config.yaml
  logs.json.txt
  run_summary.json
  best_validation.json
  best_risk_validation.json
  best_risk_validation_predictions.npz
  latest_validation.json
  checkpoints/best.ckpt
  checkpoints/best_risk.ckpt
  checkpoints/latest.ckpt
  checkpoints/epoch_0025.ckpt
  ...
  checkpoints/epoch_0500.ckpt
```

`best_risk_validation.json`包含逐任务/逐episode/逐扩散步风险、K组三元组预测MSE，以及每个去噪步的episode均衡平均RGB权重；对应NPZ保留 `[context,K,3]` 预测/标签、`[context,K]` 权重、噪声步ID。

断点继续可在训练命令后追加 `resume=true`，要求相同代码、数据和配置签名。由于本轮扩展修改了训练器/模型/缓存数据类的源码，旧A训练目录的恢复签名会变化并拒绝恢复；这不影响A已完成checkpoint的推理与既有结果，勿将B输出指向A目录。

## 6. 训练完成后的原生仿真命令

先确认训练完成500epochs并检查best_risk验证风险、逐步权重与PSD输出，再启动下面命令。使用原评测入口；不写新仿真程序。

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
  --fusion point \
  --statistics-checkpoint output/error_moments/per_step_history4_seed42/checkpoints/best_risk.ckpt \
  --eval-seed 42 --test-start-seed 1000 \
  -o output/eval_scheme_b_seed42/point -n 80 -d cuda:0
```

8环境并行、4任务各20回合，共80回合；沿用8条视频配置。批次随机数、历史reset和初始观测审计沿用五组协议。来源检查绑定B训练时专家哈希、缓存签名、20/8划分、checkpoint签名、逐步尺度、规划间隔和真实scheduler顺序。

输出 `eval_log.json`、`fusion_config.json`、`fusion_trace.json` 和原生视频。配置记录 `scheme=B`、实际噪声时间步顺序、统计模型哈希及 `radius=null`。trace保留每次规划的 `[batch,K]` RGB权重与 `[batch,K,3]` 二阶矩。原runner的总体 `avg_raw_weight` 只作K步平均显示，不能将该展示值误当成实际执行权重。

## 7. 对照与数据使用边界

路线A的既有成绩：等权62.50%（另一次搜索批次61.25%）、固定0.6为63.75%、A条件58.75%、A鲁棒62.50%。这些是已有种子上的开发/搜索结果，不能预期B一定超过它们。

先在原设置测试B，可作同协议开发对照；由于物理仿真复位存在细微差异，不宣称状态逐位完全相同。若要形成最终闭环比较，应冻结各方法，再在全新环境种子上共同测试，避免只给B新种子、其他方法引用旧分数。

原16条最终示范已用于路线A最终报告，其结果已经查看。B开发训练仍只读20/8缓存；本轮不自动读取原16条最终标签。若后续用同一16条比较B，需说明这是已查看测试集上的追加比较，不能重新称为从未影响方法开发的独立确认性测试。

路线B的效果、新颖性以及是否优于固定权重都待实验和相关研究比较，本文件不预先作结论。

## 8. 实现与检查

在原有文件上扩展：

- `model/error_moments.py`：可选num_steps，保留A的标量scale及原state_dict结构；B为K个scale与3K输出。
- `dataset/error_moments_dataset.py`：按risk_mode选择汇总标签或逐步标签，仍禁止验证等分区拟合尺度。
- `workspace/train_error_moments.py`：逐步损失、风险、checkpoint和验证报告。
- `policy/error_moment_fusion.py`：逐步矩加载、一次前向及按真实时间步查权重。
- `workspace/eval_policy_sim.py`：原CLI增加B checkpoint入口；A及五组命令继续支持。
- `config/train_error_moments_per_step.yaml`：B独立输出目录，继承原训练预算与超参数。

```bash
python -m pytest tests/test_error_moments.py tests/test_error_moment_fusion.py \
  tests/test_moment_calibration.py tests/test_fixed_weight_search.py -q
```

53项合成检查通过：包含100组PSD输出/300维head与反传、逐步标签和训练集尺度、逐步风险汇总、合成缓存两轮训练与保存/恢复、实际checkpoint加载接口的合成专家替身、每规划一次统计调用、反序真实时间步对应不同权重、两专家共享带噪动作及A的兼容。小型合成网络的单元测试不代表真实训练或仿真已执行。

## 9. 首轮真实训练与仿真结果（用户运行后补记）

前述“尚未启动”描述为实现完成时的状态。本节记录用户随后完成的实际实验，不覆盖路线A报告。B已完成500epochs；best与best_risk均为epoch31（完成第32轮），latest为epoch499。

| 方法 | 总成功次数 | 总成功率 | 开盒 | 抬马桶盖 | 开抽屉 | 取伞 |
| --- | --- | --- | --- | --- | --- | --- |
| B条件逐步权重 | 44/80 | 55.00% | 95% | 35% | 80% | 10% |
| A条件共享权重 | 47/80 | 58.75% | 100% | 50% | 80% | 5% |
| A鲁棒共享权重 | 50/80 | 62.50% | 95% | 50% | 85% | 20% |
| 等权（原五组） | 50/80 | 62.50% | 100% | 50% | 85% | 15% |
| 固定0.6（网格搜索） | 51/80 | 63.75% | 95% | 50% | 85% | 25% |

B当前未提升：比A条件少3次，比固定0.6少7次。B与A条件使用相同专家哈希与seed协议，但初始观测仅8/80逐位相同，不将这些差异解释为已排除仿真波动的统计显著结论。

验证融合epsilon风险：B最佳0.0024723114，A最佳0.0023501224，B约高5.20%。B末轮升到0.0042142114；这不是只因为最后一步闭环执行才出现的差异。A/B采用不同的输出尺度归一化，不直接比较二阶矩回归损失数值。

运行trace保存320个规划批次、2560个批次通道上下文（包括批次内已结束但仍随整批预测的通道），每个上下文100个权重。权重范围0~1；相邻扩散步平均绝对变化约0.08556；约8.48%权重落在≤0.05或≥0.95。对已保存矩阵重新代入解析公式，权重最大绝对差为0。证明记录中的权重确实逐步变化且与解析公式一致；不能仅凭波动幅度认定它就是成功率下降的原因。

进一步在相同8条开发验证轨迹、有限MC参考标签上计算不可部署的标签最优权重：共享最优风险0.0019668292，逐步最优风险0.0019456171，后者仅低约1.08%。这是基于已知参考标签的离线诊断，不是另一个可部署策略，也不是独立测试结果。它提示这批轨迹上逐步自由度的额外潜在收益较小，预测误差可能抵消收益；尚不能推广为所有任务或视触融合中逐步权重无用。

当前证据支持“这版B未获得额外收益”，不支持“B原理已被否定”。若继续研究，优先考虑降低噪声步方向的自由度或增加连续性约束，并单独验证尺度归一化因素；这些修改均尚未执行，不能根据现有最终16条结果反复调参后仍称为未查看最终测试。

来源：`output/eval_scheme_b_seed42/point/eval_log.json`、`fusion_trace.json`、`output/error_moments/per_step_history4_seed42/best_risk_validation.json` 和相应预测NPZ。分析摘要保存到 `output/error_moments/per_step_history4_seed42/post_eval_diagnostics.json`，仅由已有日志/标签计算，无新增模型推理或仿真。

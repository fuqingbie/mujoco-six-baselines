# MuJoCo 六基线复现与统一评测（MuJoCo Six-Baseline Reproduction & Unified Evaluation）

在双 H20 上复现六种鲁棒/基于模型的强化学习基线，并在**统一环境设置**下从头训练、统一冻结评测，产出两张 6 方法 × 3 任务（Hopper / HalfCheetah / Walker2d）实测成绩表。

- **阶段 A（source 协议）**：按各来源论文的协议复测/重训（含 RMBPO 作者发布权重的原样复测）
- **阶段 B（shared 协议）**：六方法在同一扰动分布（躯干质量、摩擦倍率均 U[0.90,1.10]）、同一 3,000,000 真实仿真转移预算、同一冻结评估器下统一重训（90 作业 = 6 算法 × 3 任务 × 5 种子）

## 六种算法的实现方法

| 方法 | 实现入口 | 实现方法与来源 |
|---|---|---|
| **RARL-PPO** | `mujoco_baselines/algos/rarl_ppo.py` | 移植自 [RobustRLlib@a665ddf](https://github.com/robustrl-lib/robustrllib) 的 `baselines/rarl/{game,train}.py`：双 PPO（主角+对手）1:1 交替块更新、`PlayerView` 双视角包装、`BodyForce` 对手施力（世界系 x/z，阶段 B 限 ±1N，身体部位沿用 RobustRLlib 各任务映射）。预算为双方共享计数器（各 1.5M），末块裁剪至可更新长度（SB3 RolloutBuffer 需随 n_steps 重建） |
| **RMA** | `mujoco_baselines/algos/rma.py` | 按 [SPARC@53f1e2e](https://github.com/bramgrooten/sparc) 两阶段结构移植：阶段一 QR-SAC 专家（上下文=质量/摩擦乘数），阶段二冻结专家、50 步 obs-动作历史 → 适配器表征 MSE 回归。QR-SAC 组件（公开库缺失，全 git 历史核实从未提交）按论文 §4.1 重建：32 分位、2+2 评论家、pinball-Huber 损失。网络类（ExpertPolicy/AdapterPolicy/HistoryAdapter）与 SPARC 逐位对齐 |
| **DR-PPO** | `mujoco_baselines/algos/dr_ppo.py` | SB3 PPO + 域随机化包装器重建（AD-RRL 公开库只训练 AD-RRL 本体；论文附录 E 表 4 只给区间）。阶段 B 使用统一窄分布采样器；PPO 栈与 RARL-PPO 主角完全一致以隔离 DR-vs-对手机制 |
| **QARL** | `mujoco_baselines/algos/qarl.py` | 纯 PyTorch 移植 [quantal-adversarial-rl@aae57b7](https://github.com/AryamanReddi99/quantal-adversarial-rl)（mushroom-rl 语义逐条核对）：双方在**同一**仿真转移中行动（一次转移计一次）；Gamma 温度课程（初始/目标浓度、KL≤0.5、回报门槛约束）；主角观测对手温度 α、对手温度不梯度更新（与主角的熵温度更新不同——上游语义原样保留）；评测 α=0 无对手 |
| **MBPO** | `mujoco_baselines/algos/mbpo.py`, `mbpo_core.py` | 按论文移植 Unstable-Baselines（[model_based_rl@2c2c4ac](https://github.com/x35f/model_based_rl) 恰为 [unstable_Baselines@5556a32](https://github.com/x35f/unstable_Baselines) 的子模块）：7 网 5 精英概率集成、短轨迹 rollout、UTD=2 混合回放（95% 模型/5% 真实）、SAC。与 RMBPO 共享同一骨架（唯一差异是辅助模型）；模型训练加上游计算包络上限 12000 iters/轮 |
| **RMBPO** | `mujoco_baselines/algos/rmbpo.py` | 论文（TMLR 2025）训练代码未发布——按算法 1+附录 D/G 在 MBPO 骨架上重建：辅助模型 gψ（单网 4×200 swish，输入=标准化 obs,action⊕名义模型均值/对数方差，输出=对角高斯均值/对数方差）；损失 = KL(gψ‖pθ) + η·鲁棒性 REINFORCE（价值基线归一化优势 × log-prob）；η 每任务 4/0.5/0.5；**旁路回归测试**：η=0 旁路与 MBPO 的名义 rollout 位级一致 |

## 目录结构

```
mujoco_baselines/
  constants.py           # 路径、任务、预算、种子、施力部位映射
  config.py              # 六算法冻结超参（[REBUILD] 项=偏离论文的选择，报告注明）
  grids.py               # 统一测试网格（Hopper 90 / HalfCheetah 154 / Walker2d 64 格，精确小数生成）
  doctor.py              # 环境自检（GPU/导入/网格/扰动语义/无噪声 step 复现性）
  smoke.py               # 各方法 20k 步烟测 + 评估管线验证
  verify.py              # 验收：每格从逐回合数据复算、回合数核对
  report.py              # 两张 6×3 主表聚合（种子层 bootstrap 95% CI、未见子集、低尾 10%、计算成本）
  envs/perturbed.py      # 统一扰动环境（robustness-gymnasium 语义：geom_friction 全分量×倍率、
                         #   body_mass[1] 绝对质量、pristine 拷贝不叠乘；禁用作者的 5e-3 动作噪声）
  envs/force.py          # 对手施力接口（RARL-PPO/QARL 共用）
  algos/                 # 六算法实现（见上表）
  eval/evaluator.py      # 统一冻结评估器（含末步奖励、30 复位种子 20000-20029、确定性策略、
                         #   autoreset 掩码防历史污染）
  eval/records.py        # episodes.parquet 逐回合落盘
  run/runner.py          # 作业队列（GPU 绑定、断点续跑、失败重试、僵尸进程再收割）
  run/phase_a.py         # 阶段 A 各来源 lane 的作业注册
  phase_a/               # 各来源协议驱动器（作者评估器复测/robustrllib v5/sparc 风场/AD-RRL v3 重建）
  tests/                 # RMBPO 旁路回归测试等
tools/watchdog.py        # cgroup 94 核配额下的树级并发调度 + 内存守护（82% 冻结）
configs/frozen.json      # 冻结配置快照
upstreams.lock.json      # 六上游 SHA、本地改动、环境锁
EXPERIMENT_LOG.md        # 执行日志（命令、资源实测、坑与修复链）
```

## 快速开始

```bash
# 1) 环境（三个隔离 conda env：mjb=统一 v4 栈、mjb5=robustrllib v5、sparc=gymnasium 1.2.0）
bash tools/install_envs.sh

# 2) 自检
python -m mujoco_baselines doctor

# 3) 烟测（每方法 20k 步 + 评估管线）
python -m mujoco_baselines smoke --algo rarl_ppo,dr_ppo,rma,qarl,mbpo,rmbpo

# 4) 正式队列（阶段 B 统一重训 / 阶段 A 按来源协议）
python -m mujoco_baselines run --protocol shared --workers 20 --resume
python -m mujoco_baselines run --protocol source --workers 16 --resume

# 5) 汇总与验收
python -m mujoco_baselines report full
python -m mujoco_baselines verify
```

## 评测协议要点

- 物理变动 = 官方扰动实现（robustness-gymnasium 0.29.2 fork，MuJoCo 2.3.7）：每回合 reset 从名义模型设置，整回合保持
- 主表单元 = 5 训练种子 × (格点内 30 回合平均 → 等权格点平均) → 种子平均；95% CI 种子层 bootstrap
- 每任务另报：未见子集均值（训练闭区间外）、名义条件、最差 ceil(10%×格点数) 格点均值
- 逐回合数据（protocol/algo/task/seed/condition/mass/friction/eval_seed/return/length/terminated/truncated/is_ood）可复算每个表格单元

## 关键实测结论（工程侧）

1. 小批次 SAC/PPO 在 H20 GPU+MPS 上反而比 CPU 慢 3-4×（内核 launch 延迟主导）——全 CPU 路径为实测最优
2. 容器 cgroup 实际 94 核配额（名义 384 核）→ ~40 并发作业为吞吐平台期
3. 590GB 内存限额下需内存守护防 OOM 风暴（曾一次清场全部作业，全部从检查点恢复）

详细执行记录、坑清单与修复链见 `EXPERIMENT_LOG.md`；上游版本锁定见 `upstreams.lock.json`。

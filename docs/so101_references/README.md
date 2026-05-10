# SO-101 Reference Models & Videos

HuggingFace 上扫描 100+ 个 SO-101 模型后整理的参考资料。

---

## Demo Videos (from third-party models)

> **Note:** 以下演示视频来自 HuggingFace 上其他作者的公开模型，仅作参考对比用途，非本项目训练结果。

### 1. ACT Pick Diverse Objects (real robot) — by TakuyaHiraoka

| Item | Value |
|------|-------|
| Model | `TakuyaHiraoka/act_so101_pick_diverse_objects` |
| Author | [TakuyaHiraoka](https://huggingface.co/TakuyaHiraoka) |
| Policy | ACT (Action Chunking with Transformers) |
| Downloads | 390 (SO-101 第2高) |
| Input | 腕部相机 (640x480) + 关节状态 |
| Training | 专家遥操作 + Human-in-the-Loop 纠正数据 |
| Environment | 真实 SO-101 机械臂，桌面环境 |

**Local GIF previews (from HF model assets):**

`videos/act_so101_pick_rag.gif` — 抓抹布:

![pick rag](videos/act_so101_pick_rag.gif)

`videos/act_so101_pick_pen.gif` — 抓笔:

![pick pen](videos/act_so101_pick_pen.gif)

**Source:**
- Model card: https://huggingface.co/TakuyaHiraoka/act_so101_pick_diverse_objects
- Original videos: https://huggingface.co/TakuyaHiraoka/act_so101_pick_diverse_objects/tree/main/assets

---

## Notable Models (no local video, but worth reading)

### 2. SmolVLA Pick and Place (87.66% success rate)

| Item | Value |
|------|-------|
| Model | `Sa74ll/smolvla_so101_pickandplace` |
| Policy | SmolVLA (VLA) |
| Downloads | 393 (SO-101 第1高) |
| Success Rate | 87.66% (per-joint, 5% tolerance) |
| Training | 50 episodes, position-aware stratified split |
| Key insight | 从 60.92% 提升到 87.66%，靠数据分割策略而非调参 |

Link: https://huggingface.co/Sa74ll/smolvla_so101_pickandplace

### 3. ACT Pick and Place (most liked)

| Item | Value |
|------|-------|
| Model | `AdityaRege/so101-pick-place-act` |
| Policy | ACT |
| Downloads | 121 |
| Likes | 3 (SO-101 最多) |

Link: https://huggingface.co/AdityaRege/so101-pick-place-act

### 4. DreamZero World Action Model (novel architecture)

| Item | Value |
|------|-------|
| Model | `Vizuara/dreamzero-so101-lora` |
| Policy | World Action Model (based on Wan2.1-I2V-14B) |
| Downloads | 66 |
| Likes | 3 |
| Key feature | 同时预测 24 步动作 + 33 帧未来视频 |
| Training | 400 episodes, 8 tasks |

Link: https://huggingface.co/Vizuara/dreamzero-so101-lora

### 5. SmolVLA Isaac Sim Pick Orange (detailed architecture)

| Item | Value |
|------|-------|
| Model | `edge-inference/smolvla-so101-pick-orange` |
| Policy | SmolVLA (450M total, 99.9M trainable) |
| Downloads | 126 |
| Environment | Isaac Sim (LeIsaac) |
| Architecture | Very detailed description of SmolVLA internals |

Link: https://huggingface.co/edge-inference/smolvla-so101-pick-orange

### 6. MuJoCo Sim Pick Place (similar to our pipeline)

| Item | Value |
|------|-------|
| Model | `davidlinjiahao/lerobot_so101_base_sim_pickplace` |
| Policy | ACT |
| Downloads | 1 |
| Success Rate | 66.7% @ 50K steps |
| Training | 105 demos, MuJoCo 仿真 |
| Note | Pipeline 与本项目最相似：MuJoCo sim + ACT + SO-101 |

Link: https://huggingface.co/davidlinjiahao/lerobot_so101_base_sim_pickplace

---

## Top SO-101 Models by Downloads

| # | Model | Downloads | Likes | Policy | Note |
|---|-------|-----------|-------|--------|------|
| 1 | `Sa74ll/smolvla_so101_pickandplace` | 393 | 1 | SmolVLA | 87.66% 成功率 |
| 2 | `TakuyaHiraoka/act_so101_pick_diverse_objects` | 390 | 0 | ACT | 有演示视频 |
| 3 | `open-cloth/so101-full-fold-merged-smolvla-g` | 179 | 0 | SmolVLA | |
| 4 | `Ev3Dev/so101_act_arduino_box` | 150 | 0 | ACT | |
| 5 | `wuc1/bi_so101_flatten-and-fold-the-rag-0413-model` | 141 | 1 | SmolVLA | |
| 6 | `rol09/so101-ex1-ee-z-smolvla` | 140 | 0 | SmolVLA | |
| 7 | `edge-inference/smolvla-so101-pick-orange` | 126 | 0 | SmolVLA | Isaac Sim |
| 8 | `AdityaRege/so101-pick-place-act` | 121 | 3 | ACT | 最多点赞 |
| 9 | `CursedRock17/so101_block_grab_smolvla_0` | 124 | 0 | SmolVLA | |
| 10 | `ankithreddy/pi05-so101-finetune-v2` | 125 | 0 | pi0.5 | Physical Intelligence |

---

## Key Takeaways

1. **ACT 是 SO-101 主流策略**：HF 上 30+ 个 ACT 模型，SmolVLA 约 4-5 个
2. **有演示视频的模型极少**：100+ 模型中仅 TakuyaHiraoka 有嵌入视频
3. **SmolVLA 成功率更高**：Sa74ll 的 SmolVLA 达到 87.66%（但用了更好的数据分割）
4. **我们的 pipeline 与 davidlinjiahao 最像**：都是 MuJoCo 仿真 + ACT + SO-101

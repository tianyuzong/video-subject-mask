# video-subject-mask

[English](README.md) · **中文**

在视频里找出主体、按重要性排序、分割出来，然后把掩码替换成一个通用形状，
让下游生成模型无法从掩码形状反推出主体是什么。

第二步才是重点。**精确的轮廓本身就是答案**：一个人形的洞直接告诉 diffusion transformer
"这里画个人"，它根本不需要理解画面内容，重建任务退化成描边。所以掩码会被换成一个凸原语
——凸包、旋转矩形、正矩形或椭圆——刚好覆盖主体，边缘羽化，颜色重抽，
而且每个参数都在每次运行时重新随机。

![pipeline](docs/img/pipeline_video.png)

*上：原视频。中：阶段一的精确掩码——头、手臂、衬衫下摆、两条腿都清清楚楚。
下：阶段二，一个随机颜色的通用形状。*

---

## 架构

```
                  昂贵，只跑一次                      便宜，每次推理都能重跑
  视频 ──▶ ┌──────────────────────────┐  npz  ┌───────────────────────────┐ ──▶ 遮挡视频
           │  subject_mask.py         │ ────▶ │  mask_perturb.py          │ ──▶ 软掩码
           │  GroundingDINO + SAM 2.1 │  json │  形状替换，纯 OpenCV       │ ──▶ json
           └──────────────────────────┘       └───────────────────────────┘
                457M 参数，需要 GPU                 0 参数，3 ms/rank-frame
```

拆开是有意的。分割是 457M 参数的神经网络，形状扰动是纯 OpenCV。
一条视频只分割一次，之后每个训练步花几毫秒重抽一个随机形状，
而不必重跑检测和跟踪。

---

## 安装

```bash
git clone https://github.com/tianyuzong/video-subject-mask.git
cd video-subject-mask
pip install -r requirements.txt
./fetch_models.sh          # 从 GitHub release 拉取两个检查点
```

`transformers >= 5.2` 是硬性下限——`MMGroundingDinoForObjectDetection`、
`Sam2VideoModel`、`Sam2VideoProcessor` 在更早的版本里不存在。
`ffmpeg` 和 `ffprobe` 需要在 PATH 里。
已验证环境：torch 2.12、transformers 5.2.0、opencv 4.13、numpy 2.5.2、ffmpeg 6.1.1。

模型走 release 附件而不是版本控制：933 MB 和 898 MB 远超 GitHub 单文件 100 MB 的限制，
Git LFS 免费额度（1 GB）也装不下。

## 快速开始

```bash
# 阶段一 —— 检测、排序、分割
python3 subject_mask.py --video clip.mp4 \
    --detector models/mm-gdino-swinb-hf --sam models/sam2.1-hiera-large \
    --out outputs/segmented --save-masks

# 阶段二 —— 随机化掩码形状（会处理目录下所有 *_masks.npz）
python3 mask_perturb.py --masks-dir outputs/segmented --out outputs/perturbed --workers 32
```

---

## 阶段一 —— 检测、排序、分割

```
视频 → 镜头切分 → 关键帧跑 GroundingDINO → NMS → 排序 → SAM 2 框提示
     → 双向传播 → 形态学清理 → 分层着色 → 合成
```

1. **镜头切分**用 ffmpeg 场景检测，每个镜头独立开 SAM 2 session，
   否则传播会把掩码糊过镜头切点。
2. **检测**每 `--keyframe-stride` 帧一次（默认 15）。Grounding DINO 是开放词汇的：
   它接受一个短语列表，而不是固定类别表。
3. **类别无关 NMS**，阈值 `--nms-iou`（0.55）。这一步是承重的——内置词表里
   `person`/`man`/`woman` 会同时命中同一个人，不去重的话"主体"和"次主体"
   会变成同一个人身上的两个框，被涂成两种颜色。
4. **排序**依据 `det_score^0.5 × 面积占比^0.6 × 居中度`。这不是检测置信度：
   一个小的、置信度高的、偏离中心的物体 `det_score` 很高而 `subject_score` 很低。
5. **`--min-rel-score`**（0.10）会丢掉得分低于 rank1 十分之一的 rank2+。
   没有它的话，仅仅因为画面里没别的东西，一个衣架就会被提拔成"次主体"。
6. **SAM 2 传播**——所有 rank 共用一个 session、一次传播，种子帧选的是
   前 K 个主体得分总和最高的那个关键帧。每 `--recheck-stride` 帧用新的检测结果
   复核传播掩码，IoU 低于 `--iou-reseed` 就给那个对象重新注入框并从该帧重启传播，
   其他对象凭各自的记忆继续。
7. **着色**按 rank 倒序进行，所以 rank1 永远赢得重叠像素。

![ranked](docs/img/pipeline_0kpu6VM3rZU.5.png)

*111 帧里跟踪三个主体。红绿蓝全程锁在同样的三个人身上；这里的种子帧是第 105 帧，
靠反向传播正确地铺回到第 0 帧。*

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-subjects` | 3 | 着色几个层级 |
| `--prompt` | 自动 | 给一个短语（`"person"`）来显式指定目标 |
| `--min-rel-score` | 0.10 | 设 0 关闭杂物过滤 |
| `--nms-iou` | 0.55 | 去重阈值 |
| `--keyframe-stride` | 15 | 检测器调用频率 |
| `--lossless` | 关 | 输出 ffv1/mkv，主体外像素逐位不变 |
| `--save-masks` | 关 | **阶段二必需** |

### 关于有损编码

默认的 `_labeled.mp4` 是 h264 CRF 12。在真实帧上实测：rank1 的像素只有 **1.34%**
精确等于 `(255,0,0)`，但 98.2% 在 ±2 以内——4:2:0 色度下采样会把纯红压坏，
而纯绿能保住 96.5%。**不要在 mp4 上按颜色精确匹配来反查 rank。**
用 `_masks.npz` 或 JSON，它们是无损的；或者加 `--lossless`。

---

## 阶段二 —— 随机化掩码

每个 rank、每一帧：

```
对每个连通域：  凸原语（hull | rotrect | bbox | ellipse），带抖动
各连通域求并  |  原始掩码            →  base
core   = { distanceTransform(~base) ≤ d }  →  alpha == 1
alpha  = clip(1 - distanceTransform(~core) / f, 0, 1)
```

![shapes](docs/img/shapes.png)

*同一个主体过每种原语。每个上面标了相对原掩码的面积倍数。
距离场膨胀把矩形的直角磨成了圆角，这是预期效果。*

### 两条不变量

**1. `alpha == 1` 的核心区严格包含掩码。** 否则主体像素会活到遮挡视频里被模型看见。
这由构造保证，不是事后修补：每个原语包含自己的连通域，再与掩码求并补上光栅化的缺口。

**2. 核心区不等于掩码。** 这是软掩码特有的陷阱：如果 alpha 恰好在掩码边界处开始衰减，
把 alpha 阈值化到 1.0 就能还原出精确轮廓，虚化就成了装饰。
所以渐变带完全落在扩张出来的区域里，而不透明的核心区本身是个凸原语。

![alpha](docs/img/alpha_0kpu6VM3rZU.5.png)

*每个 rank：原掩码、不透明核心区、羽化后的 alpha。
逐连通域的 solidity 从 ~0.65 升到 ~0.97——核心区是凸多边形，不是轮廓。*

### 为什么按连通域取形状，而不是整个掩码一个

在 783 个真实 rank-frame 上实测：

| | 均值 | p90 | p99 | **最大** |
|---|---|---|---|---|
| 整个掩码一个凸包 | 1.489 | 1.775 | 3.834 | **17.486** |
| 按连通域分别取 | 1.248 | 1.356 | 1.700 | **2.149** |

掩码平均有 1.9 个连通域，最多到 40 个（一堆腿）。
全局凸包会把碎片之间的空隙整个吞掉，直接涂掉半个画面。

### 为什么原语要和掩码求并

从取整后的顶点光栅化可能差一个像素。在 198 个 rank-frame 上裸测：

```
hull      覆盖 183/198      bbox      覆盖 185/198
rotrect   覆盖  63/198  ←   ellipse   覆盖 188/198
circle    覆盖  54/198  ←   triangle  覆盖  50/198  ←
```

`cv2.boxPoints` 返回浮点而 `np.int32()` 直接截断，旋转矩形四条斜边全都中招。
加上并集之后**四种形状全部 198/198**。
`_perturb_frame` 末尾那行 `alpha[mask] = 1.0` 只能修正 alpha 的值，
核心区的形状本身还是错的。

**圆形和三角形被排除在池子外**：圆形套在细长的人体上平均 2.22 倍面积、峰值 4.60 倍，
涂掉一大片背景却没带来额外的模糊性。

### 自动随机化

不需要每次调用传任何参数。按 `(视频, 镜头, rank)` 抽样：

| 参数 | 范围 | 含义 |
|---|---|---|
| `shape` | hull 40% / rotrect 25% / bbox 15% / ellipse 20% | 偏向凸包，它 1.35 倍就能覆盖，其余要 1.80–2.20 倍 |
| `shape_jitter` | 0.01–0.03 | 向外推的量，按 `sqrt(面积)` 的比例 |
| `dilate` | 0.01–0.03 | 形状之上再膨胀 |
| `feather` | 0.02–0.04 | alpha 渐变带宽度 |
| `aniso` | 1.0–1.15 | 拉伸，让团块不总是各向同性 |
| `drift_amp` / `drift_period` | 0.04–0.12 / 1.5–4.0 秒 | 缓慢的时序漂移 |
| `fill_rgb` | HSV 抽样 | 高饱和，且与同镜头内其他 rank 拉开 ≥ 90 |

调参改 `mask_perturb.py` 顶部的 `RANGES`、`SHAPES`、`SHAPE_WEIGHTS`、`SHAPE_CAPS`。
那才是设计好的调参面，CLI 有意不为每个参数开一个开关。

![seeds](docs/img/seeds_video.png)

*同一条视频、同一份掩码，四个 seed。形状和颜色都变了；
即便三次都抽到凸包，轮廓也各不相同——抖动、膨胀、羽化、各向异性都是独立抽的。*

### 掩码面积

按"覆盖即可，不要过度"调的。783 个 rank-frame 上：

```
growth 均值 1.490   p90 2.102   p99 2.417   最大 2.535

hull     n=591  均值 1.353  p99 1.839  最大 2.224
rotrect  n=138  均值 1.802  p99 2.495  最大 2.535
ellipse  n= 54  均值 2.195  p99 2.323  最大 2.323
```

收小 `shape_jitter` / `dilate` / `feather` 的收益是递减的，因为**形状本身是地板**：
从 (0.06, 0.05, 0.05) 压到 (0.01, 0.015, 0.025)，凸包也只从 1.59 倍降到 1.34 倍。
真正的杠杆是 `SHAPE_WEIGHTS`。形状多样性和紧致覆盖是此消彼长的，
默认值拿一点多样性换明显更少的背景遮挡。
四个权重设成相等可得最大多样性，只留 `hull` 则面积最小。

每种形状的上限（`SHAPE_CAPS`，被 `--max-growth` 缩放）都设在各自实测 p99 略上方。
形状是上限压不下去的地板，所以二分只会回退膨胀量——
遇到新月形连通域导致 `fitEllipse` 缩放系数爆炸的椭圆，会回退成凸包，
加这个兜底之前实测能到 9.07 倍。

### 时序行为

参数按镜头抽一次然后正弦漂移，抖动的 RNG 每帧用同一个种子重新播种，
所以某个顶点的外推量在整个镜头里保持不变。
逐帧独立随机会错两次：边界会蠕动，而且平均足够多帧就能把真实轮廓还原出来。
冒烟测试量的正是这一点——时间平均后的核心区 solidity 仍有 0.940。

### 并行安全的种子

```python
unit_seed = blake2b(f"{run_seed}|{stem}|{shot_index}|{rank}").digest()
```

不用时间也不用 PID 播种：同一毫秒启动的 worker 会碰撞，
把同一组参数发给不同的视频，静默地、成规模地出错。
哈希派生带来的性质是：worker 之间零协调、抽样天然互不相同、
`--seed N` 能让整批逐位重放（已验证：4 路并行的输出与串行逐位相同）、
单独重跑一条视频也和整批保持一致。

### 填充模式

![fills](docs/img/fills.png)

| `--fill` | 效果 |
|---|---|
| `random`（默认） | 每个主体一个高饱和颜色，从哈希链抽 |
| `grey` | `(127,127,127)`——归一化到 `[-1,1]` 后约等于 0，inpainting 的惯例 |
| `black` / `white` | 分布的两个极端，会注入一个模型必须去补偿的信号 |
| `noise` | 逐帧均匀噪声。完全没有平坦色先验，但编码慢（288 帧的片子 17.6 秒 vs 7.5 秒） |
| `ranked` | 阶段一的红/绿/蓝调色板。给人看的预览，不是条件输入 |

### 产物

| 文件 | 内容 |
|---|---|
| `<stem>_masked.mp4` | DiT 的输入：主体被填充色替换 |
| `<stem>_alpha_rank<N>.mkv` | 每个 rank 的软掩码，**无损 ffv1 灰度** |
| `<stem>_perturb.json` | 抽到的形状、颜色、参数、seed、增长比、耗时 |
| `<stem>_alpha.npz` | `(K, T, H, W)` uint8，需要 `--save-npz` |

alpha 轨道用 ffv1 是有意的：它的精确数值是条件输入，
有损编码会把平坦的核心区和渐变带一起糊掉。

---

## JSON

完整字段参考见 **[JSON_FIELDS.md](JSON_FIELDS.md)**。两个文件共同的约定：
帧号是**全局的**、从 0 开始；区间**两端都含**；框是像素坐标的 `[x0,y0,x1,y1]`；
颜色是 `[R,G,B]`；`rank` 1 是主体；`obj_id == rank`；npz 第一维索引是 `rank - 1`。

用 `shot_index` + `rank` 连接：

```python
labels  = json.load(open("clip_labels.json"))
perturb = json.load(open("clip_perturb.json"))
masks   = np.load("clip_masks.npz")["masks"]     # (K,T,H,W) bool，锐利
alpha   = np.load("clip_alpha.npz")["alpha"]     # (K,T,H,W) uint8，随机化后

subj = labels["shots"][0]["subjects"][0]
unit = next(u for u in perturb["units"]
            if u["shot_index"] == 0 and u["rank"] == subj["rank"])

r, f = subj["rank"] - 1, 55
assert masks[r, f].sum() == subj["per_frame"][f]["area_px"]    # 精确相等
assert np.all((alpha[r, f] >= 255) | ~masks[r, f])             # 核心区覆盖主体
```

有两个字段名很像，值得分清：`seed_box_xyxy` 是**检测器**在种子帧给的框，整个镜头固定；
`per_frame[].bbox_xyxy` 是**掩码**的紧致外接框，逐帧重算。

---

## 模型

| 目录 | 参数量 | fp32 | 是什么 |
|---|---|---|---|
| `models/mm-gdino-swinb-hf` | 232.81M | 933 MB | MM-GroundingDINO Swin-B，由 mmdetection 的 `grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth` 转换而来（`missing=0, unexpected=0`） |
| `models/sam2.1-hiera-large` | 224.45M | 898 MB | SAM 2.1 Hiera-Large，来自 `facebook/sam2.1-hiera-large` |
| **合计** | **457.26M** | **1831 MB** | |

### 为什么检测器里有个 BERT

检测器将近一半——232.81M 里的 120.15M——在文本侧。Grounding DINO **没有类别表**，
BERT 就是它的类别表。检查点里有个很明显的迹象：

```
bbox_head.cls_branches.0.bias    shape (1,)      ← 只有偏置，没有权重矩阵
```

分类是图像 query 与文本 token 的相似度，不是查表。输出形状印证了这一点：

```
prompt ["person","dog"]                                  → logits (1, 900, 256)
prompt ["person","dog","traffic light","fire hydrant"]   → logits (1, 900, 256)
```

两个类别还是四个，形状纹丝不动：900 是 query 数，256 是 `max_text_len`
**token 槽位数**，不是类别数。`logits[0,i,j]` 是第 i 个 query 与第 j 个文本 token
的匹配程度。这就是开放词汇的来源——换个 prompt 就检测别的东西，不用重训——
代价是每次调用都要跑一遍 BERT。

这个代价被摊薄了：检测器只在关键帧跑，一条 288 帧的片子大约调用 20 次。
逐帧的活是 SAM 2 干的，而**它**的参数 94.5% 在视觉编码器里，
每帧跑一次且被所有 rank 共用——这就是三个主体的成本几乎和一个一样的原因。

```
MM-GroundingDINO Swin-B                      SAM 2.1 Hiera-Large
  backbone.conv_encoder      87.38M (37.5%)    vision_encoder.backbone  212.15M (94.5%)
  text_backbone.encoder      85.05M (36.5%)    memory_attention.layers    5.92M ( 2.6%)
  text_backbone.embeddings   23.84M (10.2%)    mask_decoder.transformer   3.29M ( 1.5%)
  encoder.layers             21.91M ( 9.4%)    memory_encoder.*           1.30M ( 0.6%)
  decoder.layers             10.86M ( 4.7%)    vision_encoder.neck        0.55M ( 0.2%)
```

### 重新转换检测器

已经做过了，配方留着是为了可复现。`mmengine_stub.py` 伪造 `mmengine` 模块，
让 `torch.load` 不装框架也能打开那个检查点。

```bash
python3 convert_gdino_swinb.py \
    --ckpt grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth \
    --tokenizer-dir <含 bert-base-uncased 词表的目录> \
    --out models/mm-gdino-swinb-hf
```

只要 `load_state_dict` 报出任何 missing 或 unexpected 的 key，它就拒绝写出模型，
而不是产出一个悄无声息坏掉的东西。

---

## 测试

```bash
./run_cpu_smoke.sh clip.mp4 12           # 阶段一
./run_perturb_smoke.sh outputs/segmented # 阶段二
```

阶段二跑十四项检查。最近一次全量运行，4 条片子共 783 个 rank-frame：

```
1.  核心区包含掩码:                     783/783
2.  核心区不等于掩码:                   783/783
    核心区与掩码 IoU 均值:               0.694   （1.0 意味着完全没随机化）
3.  solidity 前 → 后:                   0.654 → 0.840
    逐连通域 solidity（凸度）:           0.968
    连通域未被粘连合并:                  783/783
4.  遮挡视频主体处为填充色:              480/480
    随机填充：各区域等于自己的颜色:       649/649
5.  相邻帧 alpha 平均差:                3.154 / 255
6.  时间平均后核心区 solidity:           0.940   （高 ⇒ 平均攻击失效）
7.  增长在 max(上限, 形状) 之内:         783/783
    growth 均值 1.490  p90 2.102  p99 2.417  最大 2.535
9.  同 seed 可复现:                     4/4
    并行(4) == 串行(1):                 4/4
    不同 seed 结果不同:                  4/4
10. 参数抽样两两不同:                    8/8
11. 分形状包含性:                        hull 591/591, rotrect 138/138, ellipse 54/54
12. 同镜头内最小颜色间距:                114.6   （要求 ≥ 90）
```

## 性能

| 阶段 | 硬件 | 吞吐 |
|---|---|---|
| 分割 | 1× A6000 | 288 帧 @ 720×960 约 1 分钟 |
| 分割 | CPU | 慢约 30 倍；够跑冒烟测试 |
| 形状扰动 | 1 CPU 核 | 3.0–3.7 ms 每 rank-frame |
| 形状扰动 | 32 worker | 4 条片子 8–15 秒墙钟 |

扰动在 `--work-res`（长边 512）上计算，再把 alpha 上采样回去。
全分辨率要 15.9 ms 每 rank-frame 而不是 4.0——为了本来就要被模糊掉的输出，
不值得付 4 倍代价。

### 集群提交

`run_gpu_job.sh` 通过 `sslaunch` 提交阶段一。队列一次分配整个 8 卡节点，
所以 `gpu_worker.sh` 把视频列表按 GPU 数取模分片。
单条视频只用一张卡、闲置七张——**一次提交多打包几条片子**：

```bash
./run_gpu_job.sh --dry-run clip1.mp4 clip2.mp4 ...   # 先看生成的 YAML
./run_gpu_job.sh clip1.mp4 clip2.mp4 ...
```

## 重新生成插图

```bash
python3 make_figures.py --masks-dir outputs/segmented --perturb-dir outputs/perturbed \
    --seed-dirs 11=outputs/seed_11 22=outputs/seed_22 \
    --fill-dirs grey=outputs/fill_grey noise=outputs/fill_noise \
    --out docs/img
```

---

## 已知局限

**rank 是按镜头划定的。** 镜头 A 的 rank1 不保证和镜头 B 的 rank1 是同一个物体。
跨镜头身份关联需要 ReID，没有实现。`shot_index` 能区分它们，JSON 的 `notes` 也写明了。

**碎片化的掩码凸化效果差。** 一堆腿会产生好几个互不相连的碎片；
凸包能罩住每一块，但 `SHAPE_CAPS` 会把膨胀回退，所以 solidity 只到 ~0.59，
而单个完整主体能到 ~0.98。

**重叠的原语整体上不是凸的。** 两个连通域的形状挨上时，它们的并集是个花生形。
逐连通域 solidity 均值 0.968，在那少数几帧上最低到 0.44；
视觉上看是两个凸块贴在一起，不是泄漏出来的轮廓。

**`--fill ranked` 的 alpha 渐变带会在两个 rank 交界处叠加**，
因为每个 rank 是用自己的 alpha 合成的。人工核对没问题；
作为条件输入请用逐 rank 的 alpha 轨道。

---

## 一个值得记住的 bug

第一版在合成音频时给 ffmpeg 传了 `-shortest`。遇到音频流结束得略早的片子，
输出就变成了 **288 帧的源只出 95 帧**——而且是静默的。
现在两个阶段每次写完都调 `count_frames()`，对不上就中止。
如果你要改合成那段代码，请保留这个检查。

## 许可

代码：MIT。随附权重遵循各自上游许可——MM-GroundingDINO（Apache-2.0，OpenMMLab）、
SAM 2.1（Apache-2.0，Meta）。

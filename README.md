# Ecommerce Multimodal RAG

本项目是一个可在 Windows 本地运行的多模态商品检索与 RAG 问答系统，基于 Amazon Berkeley Objects (ABO) 子集构建，支持：

- 文本搜商品
- 图片搜相似商品
- 文本 + 图片混合召回
- 基于检索结果的商品问答
- 中文查询支持

项目固定路径：

`D:\ecommerce-multimodal-rag`

## 1. 项目目标

这个项目的目标不是做一个“参数最大”的多模态系统，而是在以下现实约束下，做一个真正可落地、可启动、可演示、可继续迭代的本地系统：

- 本地 Windows 环境
- 6GB 显存级别设备
- 不依赖云服务
- 可解释的模块结构
- 每一步都能单独运行

最终产物不是单一模型，而是一条完整链路：

1. 数据下载与筛选
2. 数据清洗与统一文本表达
3. 图像向量索引
4. 文本向量索引
5. 检索与融合
6. 本地生成
7. 可交互前端

## 2. 目录结构

```text
D:\ecommerce-multimodal-rag
├─ app
│  └─ demo.py
├─ data
│  ├─ processed
│  └─ raw
│     └─ abo
├─ indexes
├─ outputs
├─ src
│  ├─ build_clip_index.py
│  ├─ build_text_index.py
│  ├─ common.py
│  ├─ config.py
│  ├─ download_abo.py
│  ├─ local_llm.py
│  ├─ pipeline.py
│  ├─ prepare_dataset.py
│  ├─ query_utils.py
│  ├─ rag_answer.py
│  └─ retrieve.py
├─ requirements.txt
├─ start_demo.bat
└─ README.md
```

## 3. 环境要求

- Windows
- Python 3.10+
- 建议使用独立虚拟环境
- 可选 GPU
  6GB 显存可运行当前版本

## 4. 安装依赖

```powershell
cd D:\ecommerce-multimodal-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5. 首次构建数据与索引

按以下顺序执行：

```powershell
cd D:\ecommerce-multimodal-rag
.venv\Scripts\activate

python src\download_abo.py
python src\prepare_dataset.py
python src\build_clip_index.py
python src\build_text_index.py
```

执行完成后会生成：

- `data\processed\products_subset.csv`
- `data\processed\products_clean.parquet`
- `data\processed\id2meta.json`
- `indexes\clip_image.index`
- `indexes\image_meta.pkl`
- `indexes\clip_text.index`
- `indexes\clip_text_meta.pkl`
- `indexes\text_bge.index`
- `indexes\text_meta.pkl`

## 6. 启动方式

### 方式 1：直接双击启动

双击：

`D:\ecommerce-multimodal-rag\start_demo.bat`

它会自动：

- 进入项目目录
- 尝试关闭旧的 7860 端口进程
- 启动本地服务
- 打开浏览器
- 进入 `http://127.0.0.1:7860`

### 方式 2：命令行启动

```powershell
cd D:\ecommerce-multimodal-rag
.venv\Scripts\activate
python app\demo.py
```

## 7. 界面使用方式

当前前端是双按钮模式：

### 1. Search

`Search` 只负责：

- 检索
- 商品卡片展示
- Top-K 表格
- Quick Summary 快速摘要

这样可以避免每次搜索都被生成阶段拖慢。

### 2. Generate Detailed Answer

`Generate Detailed Answer` 只在需要时生成更详细的回答。

这是一种刻意设计的交互拆分：

- 把“搜索”做快
- 把“生成”做成按需触发
- 让用户始终先看到结果，再决定是否深挖

## 8. 支持的输入

### 文本检索

在 `Text Query` 中输入：

- 英文商品描述
- 中文商品描述

示例：

- `red leather shoes`
- `black handbag`
- `wooden chair`
- `红色皮鞋`
- `白色运动鞋`
- `女包`

### 图片检索

在 `Query Image` 中上传一张商品图片。

### 混合检索

同时输入文本并上传图片，系统会做融合召回。

## 9. 系统当前行为

- 中文查询会先本地翻译为英文再做检索
- 英文查询直接进入检索链路
- 中文界面展示优先取原生中文字段；没有中文则回退英文
- 英文搜索结果只显示英文
- 详细回答按需单独生成
- 页面加载时会自动 warmup 主要模型

## 10. 关键模块说明

### `src\download_abo.py`

职责：

- 下载 ABO metadata
- 解析 listings
- 筛选前 10k 有效商品
- 下载对应图片

这里最关键的经验是：

- ABO listings 实际上是 `jsonl.gz`，不是单个 JSON 数组
- 如果把它当普通 JSON 读，会直接解析失败
- 原始图片元数据并不是“完整 URL”，而是 `path`

### `src\prepare_dataset.py`

职责：

- 去除缺失值
- 统一清洗字段
- 构建 `text_input`
- 生成 parquet 与元数据映射

这一步的意义不是简单清洗，而是把“结构化商品字段”压成一个适合文本向量模型理解的统一表达。

### `src\build_clip_index.py`

职责：

- 构建图像索引
- 同时构建 CLIP 文本索引

这一步承担了跨模态检索的核心能力：

- 文本搜图
- 图搜图

### `src\build_text_index.py`

职责：

- 用轻量文本模型构建文本语义索引

这一步补的是“商品文本理解能力”，它和 CLIP 文本索引不完全重复。

### `src\pipeline.py`

职责：

- 汇总多路检索结果
- 做简单加权融合

这个模块的价值在于：

- 避免单路检索偏差太大
- 让“图像语义”和“商品文本语义”互补

### `src\rag_answer.py`

职责：

- 快速摘要
- 详细回答生成

这里的关键设计不是“尽量多生成”，而是“生成只在值得时触发”。

### `app\demo.py`

职责：

- 提供面向最终演示的 UI
- 承接检索、卡片展示、摘要、详细回答

## 11. 架构与数据流

```text
User Query
   │
   ├─ Text Query ──> zh->en normalize (if needed)
   │                  │
   │                  ├─ BGE text encoder ──> text FAISS
   │                  └─ CLIP text encoder ─> image FAISS
   │
   ├─ Image Query ──> CLIP image encoder ──> image FAISS
   │
   └─ Fusion Layer ──> merged top-k results
                         │
                         ├─ Quick Summary
                         └─ Detailed Answer (optional)
```

这个架构最大的特点是：

- 检索和生成被明确拆开
- 多模态检索和生成不是强耦合
- 某一部分慢，不会拖死整条链

## 12. 设计取舍与思考沉淀

### 12.1 为什么不用更大的模型

因为这个项目的第一目标不是“极限效果”，而是“本地可用”。

在 6GB 显存和 Windows 本地环境下，如果直接堆大模型，会遇到：

- 初始化时间过长
- 显存不稳定
- 依赖复杂
- 演示体验差

所以这里的取舍是：

- 检索模型轻量化
- 生成模型轻量化
- 优先保证路径完整和体验稳定

这是一个工程判断，而不是单纯的模型判断。

### 12.2 为什么检索和生成必须分开

如果每次搜索都强制走生成，用户体验会出现两个问题：

1. 搜索本身变慢
2. 用户明明只想看结果，却被迫等待回答

因此前端改成双按钮模式是一个非常有价值的交互决策：

- `Search` 负责快速反馈
- `Generate Detailed Answer` 负责深入信息

这个设计比“所有能力一次性触发”更贴近真实产品。

### 12.3 为什么中文支持不能简单靠实时翻译展示

实践里已经验证：

- 中文查询转英文用于检索是值得的
- 但把所有结果字段再逐条翻回中文，代价很高
- 而且翻译出来的商品标题容易非常生硬，甚至错误

因此这里沉淀出的经验是：

- 查询翻译可以做
- 展示翻译要非常谨慎
- 对商品标题，错误翻译往往比保留英文更差

这类项目里，“不乱翻”本身就是一种质量策略。

### 12.4 为什么结果卡片化很重要

检索系统往往只关注“召回是否正确”，但在演示和实际使用里，展示质量同样重要。

如果结果只是：

- 一堆路径
- 一堆原始字段
- 一堆模型分数

用户不会觉得系统“好用”。

商品卡片化做的是把模型结果转成用户能快速消费的信息结构：

- 图片
- 标题
- 标签
- 卖点
- 推荐标记

这一步提升的不是模型能力，而是产品感。

### 12.5 为什么这个项目值得保留脚本化步骤

每一步都能单独运行，有三个好处：

1. 出错时更容易定位
2. 替换某一层实现时成本低
3. 项目更适合教学、展示和扩展

例如：

- 可以替换更好的文本模型重建文本索引
- 可以替换更好的图像模型重建图像索引
- 可以保留前端和整体流程不动

这种模块边界是长期可维护性的基础。

## 13. 性能观察

首次运行较慢是正常现象，原因包括：

- 模型首次加载到内存
- 中文查询需要先做翻译
- CLIP 与 BGE 双路编码
- 本地生成模型首次初始化

为了提升交互速度，当前版本已经做了这些优化：

- 模型进程内缓存
- 页面加载自动 warmup
- 中文快速摘要
- 详细回答按需单独触发
- 搜索与生成解耦
- 展示层避免重型翻译

当前可观察到的经验：

- 搜索链路最怕“额外展示翻译”
- 回答链路最怕“每次都生成”
- UI 层的产品化改造不该反向拖累搜索速度

## 14. 常见问题

### 1. 页面打不开

优先尝试双击：

`start_demo.bat`

如果端口被占用，它会自动尝试关闭旧的 `7860` 进程后再启动。

### 2. 搜索很慢

常见原因：

- 首次启动尚未 warmup 完成
- 中文查询包含翻译步骤
- 当前机器 CPU/GPU 性能有限

建议：

- 等 `Status` 显示 `Runtime warmed up...`
- 先点 `Search`
- 需要时再点 `Generate Detailed Answer`

### 3. Detailed Answer 为空或质量一般

当前版本使用轻量本地模型，优先保证本地可运行。若详细回答不稳定，系统会回退到结构化摘要。

### 4. 中文显示乱码

请确保：

- 使用 UTF-8 编码环境
- 通过项目内提供的 `start_demo.bat` 启动

## 15. 已踩过的重要坑

- ABO metadata 并不是普通 JSON 文件
- 图片 metadata 不是完整 URL，需要自己拼接
- Windows 下 subprocess 编码很容易出问题
- `transformers` 不同版本下 CLIP 接口有兼容差异
- “看起来更智能”的实时翻译，可能直接把交互拖慢到不可用
- 商品标题这种短文本，错误翻译非常影响感知质量

这些坑的共同点是：

- 很多问题不是模型问题，而是工程问题
- 很多体验问题不是检索精度问题，而是产品流程问题

## 16. 后续可演进方向

### 检索层

- 更细粒度的融合策略
- 更好的 rerank
- 更强的轻量跨模态模型

### 生成层

- 更稳定的本地小模型
- 可控模板回答
- 基于商品字段的半结构化回答

### 产品层

- 收藏与对比
- 点击商品卡查看详情
- 查询历史
- 推荐理由可视化

### 工程层

- 增量索引
- 更细的日志等级
- 更稳定的 Windows 启动器

## 17. 重新构建建议

如果你修改了数据处理或索引逻辑，建议重新执行：

```powershell
python src\prepare_dataset.py
python src\build_clip_index.py
python src\build_text_index.py
```

如果你修改了数据下载逻辑并希望重建原始子集，则重新执行：

```powershell
python src\download_abo.py
python src\prepare_dataset.py
python src\build_clip_index.py
python src\build_text_index.py
```

## 18. 总结

这个项目最重要的沉淀不是“把一个多模态 RAG 系统跑起来”，而是明确了一件事：

在本地资源受限的环境下，真正决定系统可用性的，往往不是模型规模，而是模块边界、交互拆分、速度控制、展示策略和工程取舍。

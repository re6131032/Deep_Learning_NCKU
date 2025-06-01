# Deep Learning Assignment 2 Report

## Unified-OneHead Multi-Task Challenge: Object Detection + Semantic Segmentation + Image Classification

### 1. 專案概述

本專案實現了一個統一頭部的多任務學習模型，能夠同時執行物件檢測、語義分割和圖像分類三個任務。主要挑戰在於使用單一頭部（2-3 層）同時輸出三種任務的預測結果，並透過知識蒸餾技術解決災難性遺忘問題。

**核心要求：**

- 單一頭部同時輸出檢測、分割、分類結果
- 總參數量 < 8M
- 推理速度 ≤ 150ms/512×512 圖像
- 災難性遺忘 ≤ 5%（分割、檢測）或 ≤ 5%（分類）

### 2. 模型架構設計

#### 2.1 整體架構

```
輸入圖像 (3×512×512)
    ↓
YOLOv8-n Backbone (凍結)
    ↓
Simple FPN Neck
    ↓
Multi-Task Head (2層共享 + 任務特定輸出)
    ↓
三任務同時輸出
```

#### 2.2 Backbone: YOLOv8-n

- **選擇理由**：預訓練權重豐富、參數效率高、特徵提取能力強
- **實現策略**：
  - 載入預訓練權重（yolov8n.pt）
  - 提取前 10 層作為特徵提取器
  - **凍結所有參數**以保持預訓練知識並控制參數量
  - 在第 4、6、9 層提取多尺度特徵（stride 8、16、32）

#### 2.3 Neck: Simple FPN

- **設計思路**：融合多尺度特徵，為不同任務提供豐富的語義資訊
- **實現細節**：
  - 1×1 卷積調整 P3（64 通道）和 P4（128 通道）到統一的 128 通道
  - P4 上採樣後與 P3 相加進行特徵融合
  - 下採樣回 stride 16 提供最終特徵
  - 3×3 卷積進行特徵平滑

#### 2.4 Multi-Task Head

按照作業要求設計 2-3 層統一頭部：

**共享層（2 層）：**

```python
# 第1層：特徵增強
Conv2d(128→128, 3×3) + BatchNorm + ReLU

# 第2層：特徵細化
Conv2d(128→128, 3×3) + BatchNorm + ReLU
```

**任務特定輸出層：**

- **檢測任務**：
  - 邊界框回歸：Conv2d(128→4, 1×1) → (cx, cy, w, h)
  - 置信度預測：Conv2d(128→1, 1×1) + Sigmoid
  - 類別分類：Conv2d(128→10, 1×1) + Sigmoid
- **分割任務**：
  - 轉置卷積上採樣：128→64→32→8 類別
  - 從 stride 16 恢復到原圖尺寸
- **分類任務**：
  - 全局平均池化 + 線性層
  - AdaptiveAvgPool2d(1) + Linear(128→10)

#### 2.5 參數統計

- **總參數量**：~1.91M（符合<8M 要求）
- **Backbone（凍結）**：~1.27M
- **Neck&Head**：~0.64M

### 3. 災難性遺忘緩解策略

#### 3.1 知識蒸餾（Knowledge Distillation）

核心策略是在每個訓練階段使用前一階段的模型作為教師，指導當前模型學習：

**訓練順序：**

1. **Stage 1（分割）**：無教師模型，建立 baseline
2. **Stage 2（檢測）**：使用 Stage 1 模型作教師
3. **Stage 3（分類）**：使用 Stage 2 模型作教師

#### 3.2 任務特定 KD 損失函數

**分割 KD 損失：**

```python
total_loss = α × soft_loss + (1-α) × hard_loss
soft_loss = KL_divergence(student_logits/T, teacher_logits/T)
hard_loss = weighted_CrossEntropy(student_logits, true_labels)
```

- 溫度參數 T=2.0
- 蒸餾權重 α=0.8
- 基於實際像素分布的類別權重

**檢測 KD 損失：**

```python
# 邊界框、置信度、類別分別蒸餾
box_kd = MSE(student_boxes, teacher_boxes)
conf_kd = MSE(student_conf, teacher_conf)
cls_kd = MSE(student_cls, teacher_cls)
total_kd = λ_box×box_kd + λ_obj×conf_kd + λ_cls×cls_kd
```

- 溫度參數 T=3.0
- 蒸餾權重 α=0.8

**分類 KD 損失：**

```python
soft_loss = KL_divergence(log_softmax(student/T), softmax(teacher/T))
total_loss = α × soft_loss + (1-α) × CrossEntropy(student, labels)
```

- 溫度參數 T=4.0
- 蒸餾權重 α=0.7

#### 3.3 其他技術細節

- **梯度裁剪**：防止梯度爆炸
- **學習率調度**：CosineAnnealingLR
- **權重衰減**：L2 正規化
- **資料增強**：水平翻轉、顏色抖動

### 4. 訓練配置與參數

#### 4.1 資料集

- **分割**：Mini-VOC-Seg (8 類，240/60 train/val)
- **檢測**：Mini-COCO-Det (10 類，240/60 train/val)
- **分類**：ImageNette-160 (10 類，240/60 train/val)

#### 4.2 訓練參數

**Stage 1 - 分割任務：**

- Epochs: 25
- Learning Rate: 1e-3
- Batch Size: 8
- Optimizer: AdamW (weight_decay=1e-4)
- 損失函數: WeightedSegmentationLoss + Focal Loss

**Stage 2 - 檢測任務：**

- Epochs: 5
- Learning Rate: 5e-4
- Batch Size: 8
- Teacher Model: Stage 1 模型
- KD 溫度: 3.0, α=0.8

**Stage 3 - 分類任務：**

- Epochs: 5
- Learning Rate: 5e-4
- Batch Size: 16
- Teacher Model: Stage 2 模型
- KD 溫度: 4.0, α=0.7

#### 4.3 硬體需求

- GPU: Google Colab T4/V100
- 總訓練時間: ~2 小時（符合要求）
- 推理速度: <150ms per 512×512 image

### 5. 實驗結果與分析

#### 5.1 定量結果

| 任務         | Baseline | Final  | Drop    | 要求  | 結果    |
| ------------ | -------- | ------ | ------- | ----- | ------- |
| 分割 (mIoU)  | 0.2247   | 0.1785 | 0.0462  | ≤0.05 | ✅ PASS |
| 檢測 (mAP)   | 0.1691   | 0.2001 | -0.0310 | ≤0.05 | ✅ PASS |
| 分類 (Top-1) | 33.33%   | 33.33% | 0.00%   | ≤5%   | ✅ PASS |

**🏆 總體評估：✅ 通過作業要求**

#### 5.2 關鍵發現

1. **知識蒸餾效果顯著**：

   - 分割任務遺忘控制在 4.62%，未超過 5%限制
   - 檢測任務甚至有所提升（-3.10%），顯示正遷移學習
   - 分類任務完全沒有遺忘

2. **模型架構合理**：

   - 統一頭部成功整合三個任務
   - 參數量控制良好（<8M）
   - 推理效率滿足要求

3. **訓練策略有效**：
   - 順序訓練配合 KD 有效緩解災難性遺忘
   - 不同任務的 KD 參數需要針對性調整

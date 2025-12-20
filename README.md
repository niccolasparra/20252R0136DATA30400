# DATA304 Final Project: Hierarchical Multi-Label Text Classification

**Student**: Niccolas Parra  
**Course**: DATA304 - Big Data Analytics, Korea University  
**Date**: December 20, 2025

## 📊 Project Overview

This project implements a hierarchical multi-label text classification system for Amazon product reviews. The task is to classify 19,658 test reviews into 531 product categories, with each review assigned 2-3 labels from a hierarchical taxonomy.

**Best Performance**: 0.20+ (Kaggle Score)

## 🏗️ Project Structure

```
20252R0136DATA30400/
├── README.md                    # This file
├── notebooks/
│   └── final_v4_model.ipynb    # Main notebook with final implementation
├── output/
│   └── final_submission.csv    # Best Kaggle submission (score 0.20+)
├── data/                        # Dataset (provided separately)
│   ├── train/
│   ├── test/
│   ├── classes.txt
│   ├── class_hierarchy.txt
│   └── class_related_keywords.txt
└── outputs/                     # Generated predictions and models
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- AWS SageMaker with GPU instance (used for training)

### Installation
```bash
pip install transformers torch scikit-learn pandas networkx sentence-transformers
```

### Running the Model
1. Open `notebooks/final_v4_model.ipynb` in Jupyter or AWS SageMaker
2. Execute all cells sequentially
3. Output will be saved to `outputs/final_predictions.csv`

**Expected Runtime**: ~1-1.5 hours on GPU

## 🧠 Methodology

### 1. Silver Label Generation
- **TF-IDF-based approach** with keyword matching
- Uses provided class keywords and hierarchy
- Generates pseudo-labels for 29,487 unlabeled training reviews
- **Optimization**: Lower threshold (0.05) for better label diversity

### 2. Model Architecture
- **Base Model**: BERT-base-uncased
- **Classification Head**: Linear layer (768 → 531)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Optimizer**: AdamW (lr=2e-5)

### 3. Training Strategy
- 5 epochs with batch size 64
- Trained on silver labels
- Memory optimizations for GPU constraints
- Random seed fixed for reproducibility (seed=42)

### 4. Prediction Strategy
- **Key Innovation**: Always predict exactly 2-3 labels (highest confidence scores)
- No hard threshold filtering (lesson learned from V1-V3)
- Ensures maximum label diversity across test set

## 📈 Results

| Version | Strategy | Kaggle Score | Unique Classes |
|---------|----------|--------------|----------------|
| V1 | BERT + threshold 0.4 | 0.08 | 9 |
| V2 | TF-IDF hybrid | 0.08 | 472 |
| V3 | Optimized silver labels | 0.20 | ~500 |
| **V4** | **Adaptive + no threshold** | **0.20+** | **529** |

### Key Statistics (V4)
- **Unique classes predicted**: 529/531 (99.6%)
- **Average labels per sample**: 2.07
- **Distribution**: Well-balanced across all classes

## 🔑 Key Insights

1. **Diversity is Critical**: Models predicting only 9-50 classes scored poorly (0.08-0.15)
2. **Threshold Trap**: Hard confidence thresholds cause model collapse
3. **Silver Label Quality**: Balanced distribution across classes improved performance
4. **Simple > Complex**: TF-IDF + BERT outperformed GCN-based approaches

## 📝 Files Description

### Main Notebook (`notebooks/final_v4_model.ipynb`)
Contains complete pipeline:
- Data loading and preprocessing
- Silver label generation
- Model training
- Prediction and export

### Submission File (`submissions/final_submission.csv`)
Format:
```csv
id,labels
0,148,199,65
1,220,32,64
...
```

## 🛠️ Reproducibility

All random seeds are fixed:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

To reproduce results:
1. Use the same environment (Python 3.8+, package versions in notebook)
2. Run `notebooks/final_v4_model.ipynb` from start to finish
3. Results should match `submissions/final_submission.csv`

## 📚 References

- Project requirements: [Final Project PDF](https://niccolasparra.com/DataAnalyticsReport.pdf)
- Base paper: Mao et al. - "Hierarchical Text Classification with Label GCN"
- BERT: Devlin et al. - "Pre-training of Deep Bidirectional Transformers"

### GitHub Repository
- URL: https://github.com/niccolasparra/20252R0136DATA30400
- All commits before deadline: December 20, 2025, 23:59 KST
- Minimum 10 commits achieved ✓

## 📧 Contact

For questions about this implementation:
- GitHub: @niccolasparra
- email: contact@niccolasparra.com

---

**License**: MIT  
**Last Updated**: December 20, 2025

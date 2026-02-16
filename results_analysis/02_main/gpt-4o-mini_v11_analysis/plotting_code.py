#!/usr/bin/env python3
"""
纯粹绘图代码 - 基于处理后的数据文件
数据来源: processed_data.json
运行此脚本前请确保 processed_data.json 文件存在
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载处理后的数据"""
    with open('processed_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为numpy数组
    true_labels = np.array(data['true_labels'])
    predictions = np.array(data['predictions'])
    roc_probabilities = np.array(data['roc_probabilities']) if data['roc_probabilities'] else None
    confidence_scores = np.array(data['confidence_scores']) if data['confidence_scores'] else None
    
    return {
        'true_labels': true_labels,
        'predictions': predictions,
        'roc_probabilities': roc_probabilities,
        'confidence_scores': confidence_scores,
        'model_name': data['model_name'],
        'total_samples': data['total_samples'],
        'convergent_samples': data['convergent_samples'],
        'non_convergent_samples': data['non_convergent_samples']
    }

def plot_roc_curve(data):
    """绘制ROC曲线"""
    true_labels = data['true_labels']
    roc_probabilities = data['roc_probabilities']
    
    if roc_probabilities is None or len(roc_probabilities) == 0:
        print("⚠️ 没有ROC概率数据，跳过ROC曲线")
        return None
    
    # 移除NaN值
    mask = ~np.isnan(roc_probabilities)
    true_labels_clean = true_labels[mask]
    roc_probabilities_clean = roc_probabilities[mask]
    
    if len(true_labels_clean) == 0:
        print("⚠️ 有效数据为空，跳过ROC曲线")
        return None
    
    fpr, tpr, thresholds = roc_curve(true_labels_clean, roc_probabilities_clean)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC曲线 (AUC = %.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)', fontsize=12)
    plt.ylabel('真阳性率 (TPR)', fontsize=12)
    plt.title('ROC曲线', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ ROC曲线已保存: roc_curve.png (AUC = %.3f)" % roc_auc)
    return roc_auc

def plot_confusion_matrix(data):
    """绘制混淆矩阵"""
    true_labels = data['true_labels']
    predictions = data['predictions']
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['不收敛', '收敛'],
               yticklabels=['不收敛', '收敛'])
    plt.title('混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 混淆矩阵已保存: confusion_matrix.png")
    return cm

def plot_pr_curve(data):
    """绘制精确率-召回率曲线"""
    true_labels = data['true_labels']
    roc_probabilities = data['roc_probabilities']
    
    if roc_probabilities is None or len(roc_probabilities) == 0:
        print("⚠️ 没有ROC概率数据，跳过PR曲线")
        return None
    
    # 移除NaN值
    mask = ~np.isnan(roc_probabilities)
    true_labels_clean = true_labels[mask]
    roc_probabilities_clean = roc_probabilities[mask]
    
    if len(true_labels_clean) == 0:
        print("⚠️ 有效数据为空，跳过PR曲线")
        return None
    
    precision, recall, thresholds = precision_recall_curve(true_labels_clean, roc_probabilities_clean)
    avg_precision = average_precision_score(true_labels_clean, roc_probabilities_clean)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='PR曲线 (AP = %.3f)' % avg_precision)
    
    # 添加基准线
    positive_ratio = np.mean(true_labels_clean)
    plt.axhline(y=positive_ratio, color='red', linestyle='--', 
               label='随机分类器 (AP = %.3f)' % positive_ratio)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title('精确率-召回率曲线', fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ PR曲线已保存: pr_curve.png (AP = %.3f)" % avg_precision)
    return avg_precision

def plot_confidence_distribution(data):
    """绘制置信度分数分布"""
    confidence_scores = data['confidence_scores']
    
    if confidence_scores is None or len(confidence_scores) == 0:
        print("⚠️ 没有置信度分数数据，跳过分布图")
        return {}
    
    # 统计分布
    valid_scores = confidence_scores[~np.isnan(confidence_scores)]
    if len(valid_scores) == 0:
        print("⚠️ 没有有效的置信度分数")
        return {}
    
    unique_scores, counts = np.unique(valid_scores, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_scores)))
    bars = plt.bar(unique_scores, counts, color=colors, tick_label=unique_scores)
    
    plt.xlabel('置信度分数 (0-9)', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title('置信度分数分布', fontsize=14)
    plt.xticks(range(0, 10))
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                '%d' % int(height), ha='center', va='bottom')
    
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印分布统计
    distribution = dict(zip(unique_scores.astype(int), counts))
    print("置信度分数分布:")
    for score in sorted(distribution.keys()):
        print("   %d: %d 个样本" % (score, distribution[score]))
    
    print("✅ 置信度分布图已保存: confidence_distribution.png")
    return distribution

def main():
    """主绘图函数"""
    print("=== 纯粹绘图代码 ===")
    print("正在加载处理后的数据...")
    
    try:
        data = load_data()
    except FileNotFoundError:
        print("❌ 找不到 processed_data.json 文件，请确保文件在当前目录")
        return
    except Exception as e:
        print("❌ 加载数据失败: %s" % str(e))
        return
    
    # 显示数据信息
    print("")
    print("数据信息:")
    print("模型名称: %s" % data['model_name'])
    print("总样本数: %d" % data['total_samples'])
    print("收敛样本: %d" % data['convergent_samples'])
    print("不收敛样本: %d" % data['non_convergent_samples'])
    
    # 绘制图表
    print("")
    print("=" * 50)
    print("1. 绘制ROC曲线...")
    auc_score = plot_roc_curve(data)
    
    print("")
    print("=" * 50)
    print("2. 绘制混淆矩阵...")
    cm = plot_confusion_matrix(data)
    
    print("")
    print("=" * 50)
    print("3. 绘制PR曲线...")
    ap_score = plot_pr_curve(data)
    
    print("")
    print("=" * 50)
    print("4. 绘制置信度分布...")
    confidence_dist = plot_confidence_distribution(data)
    
    print("")
    print("=" * 50)
    print("🎉 所有图表绘制完成！")
    print("生成的文件:")
    print("  - roc_curve.png")
    print("  - confusion_matrix.png")
    print("  - pr_curve.png")
    print("  - confidence_distribution.png")

if __name__ == "__main__":
    main()

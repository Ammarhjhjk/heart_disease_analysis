#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# تحميل البيانات
print("جارٍ تحميل بيانات أمراض القلب...")
df = pd.read_csv('heart_data.csv')

# تنظيف البيانات
print("\n=== تنظيف البيانات ===")
print("الأبعاد الأصلية:", df.shape)

# استبدال القيم المفقودة (الممثلة بـ '?') بقيم NaN
df = df.replace('?', np.nan)

# تحويل الأعمدة إلى نوع رقمي
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

# التعامل مع القيم المفقودة
print("القيم المفقودة قبل المعالجة:")
print(df.isnull().sum())

# حذف الصفوف ذات القيم المفقودة (طريقة بسيطة)
df = df.dropna()

print("الأبعاد بعد إزالة القيم المفقودة:", df.shape)

# تقسيم البيانات إلى متغيرات مستقلة ومتغير تابع
X = df.drop('target', axis=1)
y = df['target'].apply(lambda x: 1 if x > 0 else 0)  # تحويل إلى تصنيف ثنائي

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# توحيد النطاق
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تدريب النماذج
print("\n=== تدريب النماذج ===")

# الانحدار اللوجستي
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)

# تقييم النماذج
print("\n=== تقييم النماذج ===")

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nأداء {model_name}:")
    print(f"الدقة (Accuracy): {accuracy:.4f}")
    print(f"الدقة (Precision): {precision:.4f}")
    print(f"الاستدعاء (Recall): {recall:.4f}")
    print(f"نتيجة F1: {f1:.4f}")
    print("مصفوفة الارتباك:")
    print(cm)
    
    return accuracy, precision, recall, f1, cm

# تقييم الانحدار اللوجستي
lr_metrics = evaluate_model(y_test, lr_pred, "الانحدار اللوجستي")

# تقييم KNN
knn_metrics = evaluate_model(y_test, knn_pred, "KNN")

# مقارنة أداء النماذج
models = ['الانحدار اللوجستي', 'KNN']
accuracies = [lr_metrics[0], knn_metrics[0]]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['blue', 'green'])
plt.title('مقارنة دقة النماذج')
plt.ylabel('الدقة')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# مصفوفات الارتباك
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# مصفوفة الارتباك للانحدار اللوجستي
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('مصفوفة الارتباك - الانحدار اللوجستي')
ax1.set_xlabel('التنبؤ')
ax1.set_ylabel('الحقيقة')

# مصفوفة الارتباك لـ KNN
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt='d', cmap='Greens', ax=ax2)
ax2.set_title('مصفوفة الارتباك - KNN')
ax2.set_xlabel('التنبؤ')
ax2.set_ylabel('الحقيقة')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== الخلاصة ===")
print("من خلال تحليل بيانات أمراض القلب، يمكننا استنتاج ما يلي:")

if lr_metrics[0] > knn_metrics[0]:
    print("1. نموذج الانحدار اللوجستي حقق أداءً أفضل من حيث الدقة العامة")
else:
    print("1. نموذج KNN حقق أداءً أفضل من حيث الدقة العامة")
    
print("2. يمكن تحسين النماذج أكثر عن طريق:")
print("   - معالجة أفضل للقيم المفقودة")
print("   - ضبط معاملات النماذج")
print("   - استخدام تقنيات اختيار الميزات")
print("   - تجربة خوارزميات تصنيف أخرى")

print("\nتم حفظ المخططات البيانية في ملفات PNG في المجلد الحالي")

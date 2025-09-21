from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------- 1️⃣ Veri yükleme ----------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# ---------- 2️⃣ Temel veri analizi ----------
print("İlk 5 satır:")
print(df.head(), "\n")

print("Veri seti özet istatistikleri:")
print(df.describe(), "\n")

print("Veri seti bilgisi:")
print(df.info(), "\n")

# ---------- 3️⃣ Grafikler ----------
plt.hist(df['sepal length (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frekans")
plt.show()

plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'], cmap='viridis')
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.colorbar(label="Iris türü (target)")
plt.show()

# ---------- 4️⃣ Eğitim ve test setine ayır ----------
X = df[iris.feature_names]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 5️⃣ Modeller ----------
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC()
}

# ---------- 6️⃣ Modelleri eğit ve performanslarını yazdır ----------
accuracies = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    print(f"------ {name} ------")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\n")

# ---------- 7️⃣ Modellerin doğruluklarını görselleştir ----------
plt.figure(figsize=(8,5))
plt.bar(models.keys(), accuracies, color=['skyblue','orange','green','red'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Farklı Modellerin Accuracy Karşılaştırması")
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

# ---------- 8️⃣ Decision Tree görselleştirme ----------
dt_model = models["Decision Tree"]
plt.figure(figsize=(15,10))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Yapısı")
plt.show()

# ---------- 9️⃣ KNN Hyperparameter Tuning ----------
param_grid_knn = {'n_neighbors': list(range(1,11))}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)

print("KNN için en iyi k:", grid_knn.best_params_)
print("KNN için en iyi doğruluk (cross-validation):", grid_knn.best_score_)

# ---------- 10️⃣ Logistic Regression Hyperparameter Tuning ----------
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'newton-cg']  # liblinear çıkarıldı
}

grid_lr = GridSearchCV(LogisticRegression(max_iter=200), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)

print("Logistic Regression için en iyi parametreler:", grid_lr.best_params_)
print("Logistic Regression için en iyi doğruluk (cross-validation):", grid_lr.best_score_)

# ---------- 11️⃣ SVM Hyperparameter Tuning ----------
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly']
}

grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)

print("SVM için en iyi parametreler:", grid_svm.best_params_)
print("SVM için en iyi doğruluk (cross-validation):", grid_svm.best_score_)

# ---------- 12️⃣ En iyi modellerin test setinde doğruluklarını hesapla ----------
best_models = {
    "KNN": grid_knn.best_estimator_,
    "Decision Tree": models["Decision Tree"],
    "Logistic Regression": grid_lr.best_estimator_,
    "SVM": grid_svm.best_estimator_
}

best_accuracies = []

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    best_accuracies.append(acc)
    print(f"{name} en iyi model test doğruluğu: {acc:.2f}")

# ---------- 13️⃣ En iyi modellerin karşılaştırma grafiği ----------
plt.figure(figsize=(8,5))
plt.bar(best_models.keys(), best_accuracies, color=['skyblue','orange','green','red'])
plt.ylim(0,1)
plt.ylabel("Test Set Accuracy")
plt.title("En İyi Modellerin Test Seti Performans Karşılaştırması")
for i, v in enumerate(best_accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

# ---------- 14️⃣ Script kapanmaması için ----------
input("Programı kapatmak için Enter'a basın...")

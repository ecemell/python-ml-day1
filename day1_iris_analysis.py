from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Iris veri setini yükle
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Veri setini incele
print(df.head())
print(df.describe())
print(df.info())

# Sepal uzunluğu dağılımı
plt.hist(df['sepal length (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frekans")
plt.show()

# Sepal uzunluğu vs Sepal genişliği
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'], cmap='viridis')
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.colorbar(label="Iris türü (target)")
plt.show()

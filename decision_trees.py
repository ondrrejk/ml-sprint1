from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    random_state=42,
)

# 2. Trénink stromu
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# 3. Vizualizace stromu
plt.figure(figsize=(8, 4))
plot_tree(
    tree, filled=True, feature_names=["feature1", "feature2"], class_names=["0", "1"]
)
plt.show()

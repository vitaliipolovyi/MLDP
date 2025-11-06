# %%
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
# Алгоритми FIND-S та Candidate-Elimination (CE).
# Концепція для навчання (бінарна): "є ссавцем", визначена стовпцем 'milk' у датасеті Zoo.
# Атрибут 'milk' буде використано як ціль.

# Допоміжні функції для операцій з гіпотезами
def is_more_general_or_equal(h1, h2):
    for a, b in zip(h1, h2):
        if a == '?':
            continue
        if a == 'Ø':
            return False if b != 'Ø' else True
        if a != b:
            return False
    return True


def is_consistent(h, x):
    for hi, xi in zip(h, x):
        if hi == '?':
            continue
        if hi == 'Ø':
            return False
        if hi != xi:
            return False
    return True


# FIND-S algorithm
def find_s(instances, attributes):
    n_at = len(attributes)
    # найбільш специфічна гіпотеза
    h = ['Ø'] * n_at
    for x, label in instances:
        if label == 1:
            if h.count('Ø') == n_at and h[0] == 'Ø':
                # коли зустрічається перший позитивний приклад, ми можемо встановити h на цей приклад
                h = list(x)
            else:
                for i in range(n_at):
                    if h[i] == 'Ø':
                        h[i] = x[i]
                    elif h[i] != x[i]:
                        h[i] = '?'
    return h


# Candidate-Elimination algorithm
def candidate_elimination(training_instances, domains, attributes):
    n = len(attributes)
    # ініціалізація S та G
    S = {tuple(['Ø'] * n)}
    G = {tuple(['?'] * n)}

    def minimal_generalizations(h, x):
        # мінімальні узагальнення гіпотези h, щоб бути узгодженими з позитивним прикладом x
        res = []
        h = list(h)
        new_h = list(h)
        for i in range(n):
            if new_h[i] == 'Ø':
                new_h[i] = x[i]
            elif new_h[i] != x[i]:
                new_h[i] = '?'
        res.append(tuple(new_h))
        return res

    def minimal_specializations(h, neg_x):
        # генеруємо мінімальні спеціалізації h, які виключають neg_x
        res = []
        for i in range(n):
            if h[i] == '?':
                for val in domains[i]:
                    if val != neg_x[i]:
                        new_h = list(h)
                        new_h[i] = val
                        res.append(tuple(new_h))
            # якщо h[i] специфічний і дорівнює neg_x[i], ми не можемо спеціалізувати його далі тут
        return res

    for x, label in training_instances:
        if label == 1:  # позитивний приклад
            # видаляємо з G будь-яку гіпотезу, яка не узгоджується з x
            G = {g for g in G if is_consistent(g, x)}

            # для кожного s в S, який не узгоджується з x, видаляємо його та додаємо його мінімальні узагальнення
            S_new = set()
            for s in S:
                if is_consistent(s, x):
                    S_new.add(s)
                else:
                    gens = minimal_generalizations(s, x)
                    # залишаємо тільки ті узагальнення, які не є більш загальними, ніж деякі g в G
                    for g in G:
                        for gen in gens:
                            if is_more_general_or_equal(g, gen):
                                S_new.add(gen)
            S = S_new

            # видаляємо з S будь-яку гіпотезу, яка є більш загальною, ніж інша гіпотеза в S
            S = {s for s in S if not any((is_more_general_or_equal(s2, s) and s2 != s) for s2 in S)}

        else:  # негативний приклад
            # видаляємо з S будь-яку гіпотезу, яка не узгоджується з негативним прикладом (тобто, яка покриває neg приклад)
            S = {s for s in S if not is_consistent(s, x)}

            G_new = set()
            for g in G:
                if not is_consistent(g, x):
                    G_new.add(g)
                else:
                    # спеціалізуємо g мінімально, щоб виключити x
                    specs = minimal_specializations(g, x)
                    # залишаємо тільки ті спеціалізації, які є більш загальними, ніж деякі s в S
                    for spec in specs:
                        if any(is_more_general_or_equal(spec, s) for s in S):
                            G_new.add(spec)
            G = G_new

            # видаляємо з G будь-яку гіпотезу, яка є менш загальною, ніж інша в G
            G = {g for g in G if not any((is_more_general_or_equal(g2, g) and g2 != g) for g2 in G)}

    return S, G

# %%
cols = [
    'animal_name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic',
    'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins',
    'legs', 'tail', 'domestic', 'catsize', 'class_type'
]

zoo = pd.read_csv('./data/zoo/zoo.data', names=cols, header=0)  # header=0 бо заголовок є в файлі
print(zoo.head())

# %%
# Підготовка даних
feature_cols = [c for c in zoo.columns if c not in ['animal_name', 'class_type', 'milk']]
Xz = zoo[feature_cols].copy()
Xz = Xz.astype(str)
yz = zoo['milk'].astype(int)

# %%
# Приклади
instances = [(tuple(row), label) for row, label in zip(Xz.values, yz.values)]

# Домени
domains = [set(Xz[col]) for col in Xz.columns]

# %%
print("\\nFIND-S & Candidate-Elimination (концепція: milk==1)")
# FIND-S
h_find_s = find_s(instances, list(Xz.columns))
print("Гіпотеза FIND-S (найбільш специфічна):\n", dict(zip(Xz.columns, h_find_s)))

# %%
# Candidate-Elimination 

# Розділимо на тренувальні/тестові
X_train, X_test, y_train, y_test = train_test_split(Xz, yz, test_size=0.3, random_state=42, stratify=yz)
train_instances = [(tuple(row), label) for row, label in zip(X_train.values, y_train.values)]

# FIND-S
h_find_s = find_s(train_instances, list(Xz.columns))
print("\nFIND-S гіпотеза:")
print(dict(zip(Xz.columns, h_find_s)))

# Candidate-Elimination
S_final, G_final = candidate_elimination(train_instances, domains, list(Xz.columns))
print("\nFinal S (специфічна межа):")
for s in S_final:
    print(dict(zip(Xz.columns, s)))

print("\nFinal G (загальна межа):")
for g in G_final:
    print(dict(zip(Xz.columns, g)))

# %%
print("\nПокриття S на тестовій вибірці:")
for s in S_final:
    covered = sum(is_consistent(s, tuple(x)) for x in X_test.values)
    print(f"{dict(zip(Xz.columns, s))} → покриває {covered} з {len(X_test)} прикладів")
    
# %%

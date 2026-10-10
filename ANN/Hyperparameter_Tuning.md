
# GridSearchCV: Total Training Runs and Best Model Selection

## 1. How Many Total Trainings Happen?

Consider the following parameter grid:

```python
param_grid = {
    'model__neurons': [16, 32, 64, 128],
    'model__layers': [1, 2],
    'batch_size': [10, 20],
    'epochs': [30, 50]
}
```

First, count the different configurations:

| Parameter     | Number of choices |
| ------------- | ----------------: |
| Neurons       |                 4 |
| Hidden layers |                 2 |
| Batch size    |                 2 |
| Epochs        |                 2 |

Total configurations:

$$
4 \times 2 \times 2 \times 2 = 32
$$

You have `cv=3`, meaning each configuration is trained three times, once per fold.

$$
32 \times 3 = 96
$$

Since `refit=True` by default, GridSearchCV then trains the winning configuration one additional time on the entire training dataset.

| Training stage                      | Number of runs |
| ----------------------------------- | -------------: |
| Cross-validation                    |             96 |
| Final refit of the winner           |              1 |
| **Total model training runs** |   **97** |

This assumes all fits complete successfully. Each training run may contain 30 or 50 epochs, depending on its configuration.

## 2. How Is the Best Model Selected?

Let's simplify your grid to just three configurations. Suppose we want to choose the number of neurons, with `cv=3`.

### Configuration A: 16 neurons

| Fold | Validation accuracy |
| ---- | ------------------: |
| 1    |                 90% |
| 2    |                 80% |
| 3    |                 85% |

Average accuracy:

$$
\frac{90+80+85}{3}=85\%
$$

### Configuration B: 32 neurons

| Fold | Validation accuracy |
| ---- | ------------------: |
| 1    |                 90% |
| 2    |                 95% |
| 3    |                 85% |

Average accuracy:

$$
\frac{90+95+85}{3}=90\%
$$

### Configuration C: 64 neurons

| Fold | Validation accuracy |
| ---- | ------------------: |
| 1    |                 80% |
| 2    |                 85% |
| 3    |                 90% |

Average accuracy:

$$
\frac{80+85+90}{3}=85\%
$$

These numbers are hypothetical, just to demonstrate the process.

**Configuration B wins** because it has the highest mean cross-validation accuracy: 90%.

GridSearchCV performs this calculation automatically for all 32 configurations and selects the configuration with the highest average score because you specified `scoring='accuracy'`.

## 3. What Happens After the Winner Is Selected?

Suppose the winning configuration is:

```python
{
    'model__neurons': 32,
    'model__layers': 2,
    'batch_size': 20,
    'epochs': 50
}
```

The process is:

1. GridSearchCV finishes all 96 cross-validation training runs.
2. It compares the average validation accuracy of all 32 configurations.
3. It selects the configuration with the highest average accuracy.
4. Because `refit=True`, it creates a fresh model with those parameters and trains it on **100% of `X_train, y_train`**.
5. You use that refitted model to evaluate `X_test, y_test`.

The final model is not one of the three fold-trained models. It is a newly trained model using the winning hyperparameters and the entire training set.

You can inspect the results with:

```python
print(grid_result.best_params_)
print(grid_result.best_score_)

best_model = grid_result.best_estimator_
```

Remember:

* `best_params_` gives the winning hyperparameter combination.
* `best_score_` gives the best average cross-validation accuracy.
* `best_estimator_` gives the refitted winning model.
* Test accuracy is calculated separately using `X_test, y_test`.

**Important:** The test set must not participate in GridSearchCV. Use it only after selecting the best configuration to obtain a final evaluation of the model.

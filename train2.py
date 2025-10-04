from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_and_evaluate

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = KernelRidge()
    mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    print(f"Kernel Ridge Regressor MSE: {mse:.4f}")
from .core import evaluate_model, make_dataset, train_model


class ModelResult:
    def __init__(self, name, model, history, test_loss, test_acc):
        self.name = name
        self.model = model
        self.history = history
        self.test_loss = test_loss
        self.test_acc = test_acc


class ExperimentResult:
    def __init__(self, train_size, val_size, test_size, model_results):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.model_results = model_results

    @property
    def n_train(self):
        return self.train_size

    def get_model_result(self, name):
        for result in self.model_results:
            if result.name == name:
                return result
        available = ", ".join(result.name for result in self.model_results)
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")

    def histories(self):
        return {result.name: result.history for result in self.model_results}

    def models(self):
        return {result.name: result.model for result in self.model_results}


def run_experiment(
    model,
    train_size,
    val_size=4000,
    test_size=10000,
    model_name=None,
    epochs=120,
    batch_size=1024,
    lr=3e-3,
    weight_decay=1e-5,
    eval_batch_size=4096,
    restore_best_model=True,
    grad_clip_norm=1.0,
    use_cosine_scheduler=True,
    device="cpu",
):
    if model_name is None:
        model_name = getattr(model, "name", model.__class__.__name__)

    print("=" * 80)
    print(f"Training set size: {train_size}")

    x_train, y_train = make_dataset(train_size, device=device)
    x_val, y_val = make_dataset(val_size, device=device)
    x_test, y_test = make_dataset(test_size, device=device)

    model = model.to(device)
    history = train_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        eval_batch_size=eval_batch_size,
        restore_best_model=restore_best_model,
        grad_clip_norm=grad_clip_norm,
        use_cosine_scheduler=use_cosine_scheduler,
    )
    test_loss, test_acc = evaluate_model(
        model,
        x_test,
        y_test,
        batch_size=eval_batch_size,
    )
    print(f"[{model_name}] test loss = {test_loss:.4f}, test acc = {test_acc:.4f}")

    return ExperimentResult(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        model_results=[
            ModelResult(
                name=model_name,
                model=model,
                history=history,
                test_loss=test_loss,
                test_acc=test_acc,
            )
        ],
    )

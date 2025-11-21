import optuna
from ultralytics import YOLO

def train_yolo(model_name, lr0, batch_size, epochs, img_size):
    """
    Train YOLOv8 model with given hyperparameters.
    Returns mAP_50_95 (mean average precision) for Optuna.
    """
    model = YOLO(model_name)
    results = model.train(
        data='/kaggle/working/yolo_data/data.yaml',
        lr0=lr0,
        batch=batch_size,
        epochs=epochs,
        imgsz=img_size,
        verbose=True,
        amp=False
    )
    
    try:
        mAP50_95 = results.metrics.get('mAP_50_95', None)
        if mAP50_95 is None:
            mAP50_95 = results.box.map[-1]
    except:
        mAP50_95 = 0.0
    return mAP50_95


def objective(trial, model_name, epochs):
    lr0 = trial.suggest_loguniform('lr0', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    img_size = trial.suggest_categorical('img_size', [320, 416, 512, 640])
    
    mAP = train_yolo(
        model_name=model_name,
        lr0=lr0,
        batch_size=batch_size,
        img_size=img_size,
        epochs=epochs
    )
    return mAP


def optimize(model, n_trials, epochs):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_name=model, epochs=epochs), n_trials=n_trials)
        return study.best_params


def train_with_best_params(model_name, best_params, epochs=50):
    """
    Train a YOLO model using the given best hyperparameters.
    
    Args:
        model_name (str): The YOLO model to train.
        best_params (dict): Dictionary containing the hyperparameters.
        epochs (int): Number of training epochs.
    """
    train_yolo(
        model_name=model_name,
        lr0=best_params['lr0'],
        batch_size=best_params['batch_size'],
        img_size=best_params['img_size'],
        epochs=epochs
    )
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
    """
    Objective function for Optuna hyperparameter optimization of YOLOv8.

    This function defines the search space for hyperparameters (learning rate, 
    batch size, image size) and trains a YOLOv8 model using these parameters 
    for a given number of epochs. It returns the model's mean Average Precision 
    (mAP) as the optimization target.

    Parameters
    ----------
    trial : optuna.trial.Trial
        An Optuna trial object used to suggest hyperparameter values.
    model_name : str
        The YOLOv8 model checkpoint to use (e.g., 'yolov8s.pt', 'yolov8n.pt').
    epochs : int
        Number of epochs to train the model during this trial.

    Returns
    -------
    float
        The mean Average Precision (mAP) of the trained YOLOv8 model. 
        Optuna will attempt to maximize this value.
    """

    lr0 = trial.suggest_loguniform('lr0', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    img_size = trial.suggest_categorical('img_size', [640, 768])
    
    mAP = train_yolo(
        model_name=model_name,
        lr0=lr0,
        batch_size=batch_size,
        img_size=img_size,
        epochs=epochs
    )
    return mAP


def optimize(model, n_trials, epochs):
    """
    Optimize hyperparameters of a YOLOv8 model using Optuna.

    This function creates an Optuna study to maximize the mean Average Precision (mAP)
    of a given YOLOv8 model. It runs a specified number of trials, where each trial
    evaluates a set of hyperparameters (learning rate, batch size, image size) 
    suggested by Optuna. The best hyperparameters found across all trials are returned.

    Parameters
    ----------
    model : str
        The YOLOv8 model checkpoint to optimize (e.g., 'yolov8s.pt', 'yolov8n.pt').
    n_trials : int
        Number of Optuna trials to run for hyperparameter optimization.
    epochs : int
        Number of training epochs per trial.

    Returns
    -------
    dict
        Dictionary containing the best hyperparameters found by Optuna 
        (e.g., {'lr0': 0.01, 'batch_size': 16, 'img_size': 640}).
    """

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model_name=model, epochs=epochs), n_trials=n_trials)
    return study.best_params


def train_and_export_onnx(model_name, best_params, epochs=50):
    """
    Train YOLOv8 with best hyperparameters and export to ONNX.

    Args:
        model_name (str): YOLOv8 model checkpoint (e.g., 'yolov8s.pt' or pretrained)
        best_params (dict): Dictionary of best hyperparameters from Optuna
        epochs (int): Number of training epochs
        imgsz (int): Image size for training
        onnx_path (str): Path to save ONNX model
    """

    model = YOLO(model_name)
    
    model.train(
        data='/kaggle/working/yolo_data/data.yaml',
        lr0=best_params['lr0'],
        batch=best_params['batch_size'],
        imgsz=best_params['img_size'],
        epochs=epochs
    )
    
    model.export(format='onnx', opset=12, simplify=True)


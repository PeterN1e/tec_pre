import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json


class TECPredictor:
    def __init__(self,
                 model: torch.nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        TEC预测器

        Args:
            model: 训练好的Transformer模型
            device: 运行设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict_single(self,
                       tec_features: np.ndarray,
                       meta_data: np.ndarray) -> np.ndarray:
        """
        单次预测

        Args:
            tec_features: [seq_length, input_dim] 输入TEC特征
            meta_data: [seq_length, meta_dim] 输入元数据

        Returns:
            [pred_length, input_dim] 预测结果
        """
        # 转换为tensor
        tec_tensor = torch.tensor(tec_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        meta_tensor = torch.tensor(meta_data, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(tec_tensor, meta_tensor)

        return prediction.squeeze(0).cpu().numpy()

    def predict_batch(self,
                      tec_features_batch: np.ndarray,
                      meta_data_batch: np.ndarray) -> np.ndarray:
        """
        批量预测

        Args:
            tec_features_batch: [batch_size, seq_length, input_dim] 批量TEC特征
            meta_data_batch: [batch_size, seq_length, meta_dim] 批量元数据

        Returns:
            [batch_size, pred_length, input_dim] 预测结果
        """
        tec_tensor = torch.tensor(tec_features_batch, dtype=torch.float32).to(self.device)
        meta_tensor = torch.tensor(meta_data_batch, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model(tec_tensor, meta_tensor)

        return predictions.cpu().numpy()

    def rolling_prediction(self,
                           tec_features: np.ndarray,
                           meta_data: np.ndarray,
                           future_meta: Optional[np.ndarray] = None,
                           forecast_horizon: int = 24,
                           step: int = 6) -> np.ndarray:
        """
        滚动预测

        Args:
            tec_features: [seq_length, input_dim] 初始TEC特征
            meta_data: [seq_length, meta_dim] 历史元数据
            future_meta: [forecast_horizon, meta_dim] 未来元数据（可选）
            forecast_horizon: 总预测长度
            step: 每次预测的步长

        Returns:
            [forecast_horizon, input_dim] 预测结果
        """
        all_predictions = []
        current_tec = tec_features.copy()
        seq_length = tec_features.shape[0]

        for i in range(0, forecast_horizon, step):
            current_meta = meta_data[-seq_length:]

            # 如果有未来元数据，使用相应部分
            if future_meta is not None:
                current_future_meta = future_meta[i:i + step]

            # 预测
            pred = self.predict_single(current_tec, current_meta)

            # 取实际需要的预测长度
            actual_step = min(step, forecast_horizon - i)
            all_predictions.append(pred[:actual_step])

            # 更新输入序列
            current_tec = np.concatenate([current_tec[step:], pred[:step]])

            # 更新元数据
            if future_meta is not None:
                meta_data = np.concatenate([meta_data[step:], current_future_meta])

        return np.concatenate(all_predictions)

    def evaluate_forecast(self,
                          tec_features: np.ndarray,
                          meta_data: np.ndarray,
                          target: np.ndarray,
                          future_meta: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估预测结果

        Args:
            tec_features: 输入TEC特征
            meta_data: 输入元数据
            target: 真实目标值
            future_meta: 未来元数据（可选）

        Returns:
            评估指标字典
        """
        prediction = self.predict_single(tec_features, meta_data)

        # 计算各种指标
        mse = np.mean((prediction - target) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(prediction - target))
        mape = np.mean(np.abs((prediction - target) / (target + 1e-8))) * 100

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }

    def load_model(self, model_path: str):
        """
        加载模型权重

        Args:
            model_path: 模型权重文件路径
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print(f"Loaded model from {model_path}")

    def save_predictions(self,
                         predictions: np.ndarray,
                         filename: str,
                         metadata: Optional[Dict] = None):
        """
        保存预测结果

        Args:
            predictions: 预测结果
            filename: 保存文件名
            metadata: 额外的元数据（可选）
        """
        data = {
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat(),
            'shape': predictions.shape
        }

        if metadata:
            data.update(metadata)

        # 创建目录
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # 保存为JSON文件
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved predictions to {filename}")

    def load_predictions(self, filename: str) -> Tuple[np.ndarray, Dict]:
        """
        加载预测结果

        Args:
            filename: 预测结果文件路径

        Returns:
            (预测结果数组, 元数据字典)
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        predictions = np.array(data.pop('predictions'))
        return predictions, data


class TECVisualizer:
    """TEC预测结果可视化工具"""

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_prediction_comparison(self,
                                   actual: np.ndarray,
                                   predicted: np.ndarray,
                                   timestamps: Optional[List[datetime]] = None,
                                   title: str = 'Actual vs Predicted TEC',
                                   xlabel: str = 'Time',
                                   ylabel: str = 'TEC (TECU)',
                                   save_path: Optional[str] = None):
        """
        绘制预测对比图

        Args:
            actual: 实际值
            predicted: 预测值
            timestamps: 时间戳列表
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)

        if timestamps is None:
            timestamps = range(len(actual))

        plt.plot(timestamps, actual, label='Actual', marker='o', markersize=3, linewidth=2)
        plt.plot(timestamps, predicted, label='Predicted', marker='x', markersize=3, linewidth=2)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if len(timestamps) > 24:
            plt.xticks(rotation=45)
            plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        return plt

    def plot_error_series(self,
                          actual: np.ndarray,
                          predicted: np.ndarray,
                          timestamps: Optional[List[datetime]] = None,
                          title: str = 'Prediction Error',
                          save_path: Optional[str] = None):
        """
        绘制误差序列

        Args:
            actual: 实际值
            predicted: 预测值
            timestamps: 时间戳列表
            title: 图表标题
            save_path: 保存路径
        """
        error = actual - predicted

        plt.figure(figsize=self.figsize)

        if timestamps is None:
            timestamps = range(len(actual))

        plt.plot(timestamps, error, label='Error', color='red', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        plt.fill_between(timestamps, error, 0, where=error >= 0, color='red', alpha=0.2)
        plt.fill_between(timestamps, error, 0, where=error < 0, color='blue', alpha=0.2)

        plt.xlabel('Time')
        plt.ylabel('Error (TECU)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if len(timestamps) > 24:
            plt.xticks(rotation=45)
            plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt

    def plot_metrics_evaluation(self,
                                metrics: Dict[str, float],
                                title: str = 'Model Performance Metrics',
                                save_path: Optional[str] = None):
        """
        绘制评估指标

        Args:
            metrics: 评估指标字典
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))

        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        bars = plt.bar(metric_names, metric_values, color='skyblue')

        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom')

        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title(title)
        plt.grid(True, axis='y', alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt

    def plot_multistep_prediction(self,
                                  actual: np.ndarray,
                                  predictions: List[np.ndarray],
                                  horizons: List[int],
                                  timestamps: Optional[List[datetime]] = None,
                                  title: str = 'Multi-step Prediction',
                                  save_path: Optional[str] = None):
        """
        绘制多步预测结果

        Args:
            actual: 实际值
            predictions: 不同步长的预测结果列表
            horizons: 预测步长列表
            timestamps: 时间戳列表
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=self.figsize)

        if timestamps is None:
            timestamps = range(len(actual))

        # 绘制实际值
        plt.plot(timestamps, actual, label='Actual', color='black', linewidth=2)

        # 绘制不同步长的预测
        colors = plt.cm.jet(np.linspace(0, 1, len(predictions)))

        for i, (pred, horizon) in enumerate(zip(predictions, horizons)):
            plt.plot(timestamps[:len(pred)], pred,
                     label=f'Pred {horizon}h',
                     color=colors[i],
                     linewidth=1.5,
                     alpha=0.8)

        plt.xlabel('Time')
        plt.ylabel('TEC (TECU)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if len(timestamps) > 24:
            plt.xticks(rotation=45)
            plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return plt


def create_time_series_plot(actual: np.ndarray,
                            predicted: np.ndarray,
                            timestamps: List[datetime],
                            title: str = 'TEC Prediction',
                            save_dir: str = 'results',
                            filename: str = 'prediction_plot.png'):
    """
    创建时间序列预测图

    Args:
        actual: 实际值
        predicted: 预测值
        timestamps: 时间戳
        title: 图表标题
        save_dir: 保存目录
        filename: 文件名
    """
    visualizer = TECVisualizer()

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 绘制预测对比图
    plt = visualizer.plot_prediction_comparison(
        actual=actual,
        predicted=predicted,
        timestamps=timestamps,
        title=title,
        save_path=save_path
    )

    plt.close()


def analyze_prediction_results(predictions: np.ndarray,
                               targets: np.ndarray,
                               timestamps: List[datetime],
                               save_dir: str = 'results'):
    """
    分析预测结果并生成报告

    Args:
        predictions: 预测结果
        targets: 目标值
        timestamps: 时间戳
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 计算评估指标
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100

    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }

    # 保存指标
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Prediction Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")

    # 生成可视化
    visualizer = TECVisualizer()

    # 预测对比图
    visualizer.plot_prediction_comparison(
        actual=targets,
        predicted=predictions,
        timestamps=timestamps,
        title='TEC Prediction Results',
        save_path=os.path.join(save_dir, 'prediction_comparison.png')
    )

    # 误差序列图
    visualizer.plot_error_series(
        actual=targets,
        predicted=predictions,
        timestamps=timestamps,
        title='Prediction Error Analysis',
        save_path=os.path.join(save_dir, 'error_series.png')
    )

    # 指标图
    visualizer.plot_metrics_evaluation(
        metrics=metrics,
        title='Model Performance Metrics',
        save_path=os.path.join(save_dir, 'metrics_plot.png')
    )

    plt.close('all')
    print(f"Analysis results saved to {save_dir}")
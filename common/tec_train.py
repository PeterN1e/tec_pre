import torch
import torch.nn as nn
from config import TrainConfig,DatasetConfig,ModelConfig

cfg_train = TrainConfig
cfg_dataset = DatasetConfig
cfg_model = ModelConfig

import os  #处理文件和目录
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging  #跟踪程序的运行状态、调试错误以及记录重要信息
# 导入进度条模块，tqdm可以让我们在训练过程中看到每个epoch的进度
from tqdm import tqdm
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(
    level=logging.INFO,#只有 INFO 级别及以上（INFO、WARNING、ERROR、CRITICAL）的日志会被处理；DEBUG 会被忽略。
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',#依次是时间、logger 名、级别、正文。
    handlers=[                                  # 同时把日志送到两个地方
        logging.FileHandler(cfg_train.log_path),  #所保存的日志
        logging.StreamHandler()               #控制台输出，可以实时查看进度
    ]
)
logger = logging.getLogger(__name__)

class TrainModel:
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 save_best
                 ):
        super().__init__()
        """
        Args:
        model: PyTorch 模型
        optimizer: 优化器
        criterion: 损失函数
        train_loader: 训练 DataLoader
        test_loader: 测试 DataLoader
        device: 计算设备
        scheduler: 学习率调度器（可选），由外部创建并传入
        patience: 早停的耐心值（连续多少个 epoch 验证损失不下降则停止）
        save_best: 是否保存验证损失最低的模型
        model_save_path: 最佳模型保存路径
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = cfg_train.batch_size
        self.model_name = cfg_model.model_name
        self.seq_length = cfg_train.seq_length
        self.pred_length = cfg_train.pred_length
        self.dataset_year = cfg_dataset.dataset_year
        self.device = cfg_train.device
        self.scheduler = scheduler,
        self.save_best = save_best
        self.patience = 5
        self.best_test_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def train(self,num_epochs):
        train_losses = []
        test_losses=[]
        logger.info(f'------batch_size {self.batch_size:3d}| '
                    f'model: {self.model_name}| '
                    f'seq_length:{self.seq_length:3d}  pred_length:{self.pred_length:3d}'
                    f'所用数据集:{self.dataset_year:3d}'
                    )

        for epoch in range(1,num_epochs+1):
            self.model.train()#模型测试模式
            train_loss ,num= 0.0,0
            pbar = tqdm(self.train_loader,
                        total=len(self.train_loader),
                        ncols=100,
                        desc=f'Epoch {epoch}/{num_epochs}',
                        leave=False)
            for batch_in_tec,batch_in_aux,batch_exp_tec,batch_exp_aux in pbar:

                batch_in_tec = batch_in_tec.float().to(self.device)#转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                batch_in_aux = batch_in_aux.float().to(self.device)
                batch_exp_tec = batch_exp_tec.float().to(self.device)
                batch_exp_aux = batch_exp_aux.float().to(self.device)
                #batch_exp(24,71,73)
                output = self.model(batch_in_tec,batch_in_aux)

                loss = self.criterion(output, batch_exp_tec)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()
                avg_loss = train_loss / (pbar.n + 1)
                pbar.set_postfix({'batch_loss':f'{loss.item():.4f}',
                                  'avg':f'{avg_loss:.4f}'})

            ###############验证阶段##############
            self.model.eval()  # 模型切换为评估模式
            test_loss = 0.0
            with torch.no_grad():
                for batch_in_tec,batch_in_aux,batch_exp in self.test_loader:

                    batch_in_tec = batch_in_tec.float().to(self.device)  # 转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                    batch_in_aux = batch_in_aux.float().to(self.device)
                    batch_exp_tec = batch_exp_tec.float().to(self.device)
                    outputs = self.model(batch_in_tec,batch_in_aux)
                    test_loss += self.criterion(outputs, batch_exp_tec).item()
            avg_train_loss = train_loss / len(self.train_loader)
            avg_test_loss = test_loss / len(self.test_loader)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            logger.info(f'Epoch {epoch:3d} | '
                        f'Train {avg_train_loss:.5f} | '
                        f'Test  {avg_test_loss:.5f}')
            pbar.close()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_test_loss)
                else:
                    self.scheduler.step()  # 其他调度器（StepLR, CosineAnnealingLR 等）

            if avg_test_loss < self.best_test_loss:
                self.best_test_loss = avg_test_loss
                self.counter = 0
                if self.save_best:
                    torch.save(self.model.state_dict(), self.model_save_path)
                    logger.info(f'Best model saved at epoch {epoch} with test loss {avg_test_loss:.5f}')
        return train_losses, test_losses

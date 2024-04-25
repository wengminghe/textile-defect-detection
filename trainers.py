import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np

from datasets.textile_dataset import TextileDatasetStage1, TextileDatasetStage2
from models.msflow.msflow import MSFlow
from models.msflow.loss import MSFlowLoss
from models.model import Model, Model0
from utils.metric import accuracy_precision_recall1, accuracy_precision_recall2


class Trainer:
    def __init__(self, configs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = configs['mode']
        self.epoch = 1
        self.epochs = configs['train']['epochs']
        self.exp_name = configs['name']

        self.output_dir = os.path.join(configs['train']['output_dir'], configs['name'])

        if self.mode == 'train':
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir)

        self.train_dataloader = self._train_dataloader(configs['dataset'])
        self.val_dataloader = self._val_dataloader(configs['dataset'])
        self.test_dataloader = self._test_dataloader(configs['dataset'])

        self.model = Model0(**configs['model']).to(self.device)
        self.optimizer = torch.optim.Adam(self._trainable_parameters(), **configs['train']['optim'])

        self.accuracy = -1
        self.precision = -1
        self.recall = -1
        self.loss = 1e5
        self.load_checkpoint()

    @staticmethod
    def _train_dataloader(config):
        dataset = TextileDatasetStage2(**config, split='train')
        dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'], pin_memory=True)
        return dataloader

    @staticmethod
    def _val_dataloader(config):
        dataset = TextileDatasetStage2(**config, split='val')
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'], pin_memory=True)
        return dataloader

    @staticmethod
    def _test_dataloader(config):
        dataset = TextileDatasetStage2(**config, split='test')
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'], pin_memory=True)
        return dataloader

    def _trainable_list(self):
        return [self.model.fusion, self.model.out]

    def _trainable_parameters(self):
        params = []
        for m in self._trainable_list():
            params += list(m.parameters())
        return params

    def training(self):
        for module in self._trainable_list():
            module.train()

    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.output_dir, 'checkpoint.pth')):
            data = torch.load(os.path.join(self.output_dir, 'checkpoint.pth'), map_location='cpu')
            self.model.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
            self.epoch = data['epoch']
            self.accuracy = data['accuracy']
            self.precision = data['precision']
            self.recall = data['recall']
            self.loss = data['loss']

    def save_checkpoint(self):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'loss': self.loss,
        }
        torch.save(data, os.path.join(self.output_dir, 'checkpoint.pth'))

    def save_model(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, name))

    def run(self):
        # self.val_epoch()
        if self.mode == 'test':
            state_dict = torch.load(os.path.join(self.output_dir, 'best_recall_model.pth'))
            self.model.load_state_dict(state_dict)
            self.test_epoch()
            return
        while self.epoch <= self.epochs:
            self.train_epoch()
            self.val_epoch()
            self.test_epoch()
            self.epoch += 1
            self.save_checkpoint()

    def train_epoch(self):
        self.training()

        epoch_loss = 0.
        p_bar = tqdm(self.train_dataloader, desc=f'<{self.exp_name}> Train Epoch[{self.epoch}/{self.epochs}]')
        for image, label in p_bar:
            image = image.to(self.device)
            label = label.to(self.device)
            logits = self.model(image)
            loss = F.cross_entropy(logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            p_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss /= len(self.train_dataloader)
        print(f'Train Epoch: [{self.epoch}/{self.epochs}] Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar('train/loss', epoch_loss, self.epoch)

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        label_list = []
        pred_list = []
        epoch_loss = 0.

        p_bar = tqdm(self.val_dataloader, desc=f'<{self.exp_name}> Val Epoch[{self.epoch}/{self.epochs}]')
        for image, label in p_bar:
            image = image.to(self.device)
            label = label.to(self.device)
            label_list.extend(label.cpu().numpy())

            logits = self.model(image)
            loss = F.cross_entropy(logits, label)
            epoch_loss += loss.item()
            pred = self.model.predict(image)
            pred_list.append(pred)

        epoch_loss /= len(self.val_dataloader)
        accuracy, precision, recall = accuracy_precision_recall2(pred_list, label_list)
        print(f'Val Epoch: [{self.epoch}/{self.epochs}] Loss: {epoch_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

        if self.mode == 'train':
            self.writer.add_scalar('val/loss', epoch_loss, self.epoch)
            self.writer.add_scalar('val/accuracy', accuracy, self.epoch)
            self.writer.add_scalar('val/precision', precision, self.epoch)
            self.writer.add_scalar('val/recall', recall, self.epoch)

            if accuracy > self.accuracy:
                self.save_model('best_accuracy_model.pth')
                self.accuracy = accuracy
            if precision > self.precision:
                self.save_model('best_precision_model.pth')
                self.precision = precision
            if recall > self.recall:
                self.save_model('best_recall_model.pth')
                self.recall = recall
            if epoch_loss < self.loss:
                self.save_model('best_loss_model.pth')
                self.loss = epoch_loss

    @torch.no_grad()
    def test_epoch(self):
        self.model.eval()
        label_list = []
        pred_list = []

        p_bar = tqdm(self.test_dataloader, desc=f'<{self.exp_name}> Test Epoch[{self.epoch}/{self.epochs}]')
        for image, label in p_bar:
            image = image.to(self.device)
            label = label.to(self.device)
            label_list.extend(label.cpu().numpy())

            pred = self.model.predict(image)
            pred_list.append(pred)

        accuracy, precision, recall = accuracy_precision_recall2(pred_list, label_list)
        print(f'Test Epoch: [{self.epoch}/{self.epochs}]'
              f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

        if self.mode == 'train':
            self.writer.add_scalar('test/accuracy', accuracy, self.epoch)
            self.writer.add_scalar('test/precision', precision, self.epoch)
            self.writer.add_scalar('test/recall', recall, self.epoch)

class TrainerStage1:
    def __init__(self, configs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = configs['mode']
        self.epoch = 1
        self.epochs = configs['train']['epochs']
        self.top_k = configs['eval']['top_k']
        self.threshold = configs['eval']['threshold']
        self.exp_name = configs['name']

        self.output_dir = os.path.join(configs['train']['output_dir'], configs['name'])

        if self.mode == 'train':
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir)

        self.train_dataloader = self._train_dataloader(configs['dataset'])
        self.val_dataloader = self._val_dataloader(configs['dataset'])

        self.model = MSFlow(**configs['model']).to(self.device)
        self.criterion = MSFlowLoss()
        self.optimizer = torch.optim.Adam(self._trainable_parameters(), **configs['train']['optim'])

        self.accuracy = -1
        self.precision = -1
        self.recall = -1
        self.loss = 1e5
        self.load_checkpoint()

    @staticmethod
    def _train_dataloader(config):
        dataset = TextileDatasetStage1(**config, split='train')
        dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'], pin_memory=True)
        return dataloader

    @staticmethod
    def _val_dataloader(config):
        dataset = TextileDatasetStage1(**config, split='val')
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'], pin_memory=True)
        return dataloader

    def _trainable_list(self):
        return [self.model.fusion_flow, self.model.parallel_flows]

    def _trainable_parameters(self):
        params = []
        for m in self._trainable_list():
            params += list(m.parameters())
        return params

    def training(self):
        for module in self._trainable_list():
            module.train()

    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.output_dir, 'checkpoint.pth')):
            data = torch.load(os.path.join(self.output_dir, 'checkpoint.pth'), map_location='cpu')
            data['model'] = self.convert_state_dict(data['model'])
            self.model.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
            self.epoch = data['epoch']
            self.accuracy = data['accuracy']
            self.precision = data['precision']
            self.recall = data['recall']
            self.loss = data['loss']

    def convert_state_dict(self, state_dict):
        maps = {}
        for i in range(len(self.model.parallel_flows)):
            maps[self.model.fusion_flow.module_list[i].perm.shape[0]] = i
        temp = {}
        for k, v in state_dict.items():
            if 'fusion_flow' in k and 'perm' in k:
                temp[k.replace(k.split('.')[2], str(maps[v.shape[0]]))] = v
        for k, v in temp.items():
            state_dict[k] = v
        return state_dict

    def save_checkpoint(self):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'loss': self.loss,
        }
        torch.save(data, os.path.join(self.output_dir, 'checkpoint.pth'))

    def save_model(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, name))

    def run(self):
        # self.val_epoch()
        if self.mode == 'test':
            state_dict = torch.load(os.path.join(self.output_dir, 'best_recall_model.pth'))
            state_dict = self.convert_state_dict(state_dict)
            self.model.load_state_dict(state_dict)
            self.val_epoch()
            return
        while self.epoch <= self.epochs:
            self.train_epoch()
            self.val_epoch()
            self.epoch += 1
            self.save_checkpoint()

    def train_epoch(self):
        self.training()

        epoch_loss = 0.
        p_bar = tqdm(self.train_dataloader, desc=f'<{self.exp_name}> Train Epoch[{self.epoch}/{self.epochs}]')
        for image, label in p_bar:
            image = image.to(self.device)
            z_list, jac = self.model(image)
            loss = self.criterion(z_list, jac)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            p_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss /= len(self.train_dataloader)
        print(f'Train Epoch: [{self.epoch}/{self.epochs}] Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar('train/loss', epoch_loss, self.epoch)

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        label_list = []
        score_list = []
        epoch_loss = 0.
        size = 0.

        p_bar = tqdm(self.val_dataloader, desc=f'<{self.exp_name}> Val Epoch[{self.epoch}/{self.epochs}]')
        for image, label in p_bar:
            image = image.to(self.device)
            label_list.extend(label.cpu().numpy())

            if label == 0:
                z_list, jac = self.model(image)
                loss = self.criterion(z_list, jac)
                epoch_loss += loss.item()
                size += 1

            score, _ = self.model.predict(image, top_k=self.top_k)
            score = np.max(score)
            score_list.append(score)

        epoch_loss /= size
        accuracy, precision, recall, threshold = accuracy_precision_recall1(score_list, label_list)
        print(f'Val Epoch: [{self.epoch}/{self.epochs}] Loss: {epoch_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, Threshold: {threshold:.3f}')

        if self.mode == 'train':
            self.writer.add_scalar('val/loss', epoch_loss, self.epoch)
            self.writer.add_scalar('val/accuracy', accuracy, self.epoch)
            self.writer.add_scalar('val/precision', precision, self.epoch)
            self.writer.add_scalar('val/recall', recall, self.epoch)
            self.writer.add_scalar('val/threshold', threshold, self.epoch)

            if accuracy > self.accuracy:
                self.save_model('best_accuracy_model.pth')
                self.accuracy = accuracy
            if precision > self.precision:
                self.save_model('best_precision_model.pth')
                self.precision = precision
            if recall > self.recall:
                self.save_model('best_recall_model.pth')
                self.recall = recall
            if epoch_loss < self.loss:
                self.save_model('best_loss_model.pth')
                self.loss = epoch_loss


class TrainerStage2:
    def __init__(self, configs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = configs['mode']
        self.epoch = 1
        self.epochs = configs['train']['epochs']
        self.exp_name = configs['name']

        self.output_dir = os.path.join(configs['train']['output_dir'], configs['name'])

        if self.mode == 'train':
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.output_dir)

        self.train_dataloader = self._train_dataloader(configs['dataset'])
        self.val_dataloader = self._val_dataloader(configs['dataset'])

        self.model = Model(**configs['model']).to(self.device)
        state_dict = torch.load(configs['model_path'], map_location='cpu')
        state_dict = self.convert_state_dict(state_dict)
        self.model.msflow.load_state_dict(state_dict, strict=False)
        self.optimizer = torch.optim.Adam(self._trainable_parameters(), **configs['train']['optim'])

        self.accuracy = -1
        self.precision = -1
        self.recall = -1
        self.loss = 1e5
        self.load_checkpoint()

    @staticmethod
    def _train_dataloader(config):
        dataset = TextileDatasetStage2(**config, split='train')
        dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                                shuffle=True, num_workers=config['num_workers'], pin_memory=True)
        return dataloader

    @staticmethod
    def _val_dataloader(config):
        dataset = TextileDatasetStage2(**config, split='val')
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=config['num_workers'], pin_memory=True)
        return dataloader

    def _trainable_list(self):
        return [self.model.fusion, self.model.out]

    def _trainable_parameters(self):
        params = []
        for m in self._trainable_list():
            params += list(m.parameters())
        return params

    def training(self):
        for module in self._trainable_list():
            module.train()

    def load_checkpoint(self):
        if os.path.exists(os.path.join(self.output_dir, 'checkpoint.pth')):
            data = torch.load(os.path.join(self.output_dir, 'checkpoint.pth'), map_location='cpu')
            data['model'] = self.convert_state_dict(data['model'])
            self.model.load_state_dict(data['model'])
            self.optimizer.load_state_dict(data['optimizer'])
            self.epoch = data['epoch']
            self.accuracy = data['accuracy']
            self.precision = data['precision']
            self.recall = data['recall']
            self.loss = data['loss']

    def convert_state_dict(self, state_dict):
        maps = {}
        for i in range(len(self.model.msflow.parallel_flows)):
            maps[self.model.msflow.fusion_flow.module_list[i].perm.shape[0]] = i
        temp = {}
        for k, v in state_dict.items():
            if 'fusion_flow' in k and 'perm' in k:
                temp[k.replace(k.split('.')[-2], str(maps[v.shape[0]]))] = v
        for k, v in temp.items():
            state_dict[k] = v
        return state_dict

    def save_checkpoint(self):
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'loss': self.loss,
        }
        torch.save(data, os.path.join(self.output_dir, 'checkpoint.pth'))

    def save_model(self, name):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, name))

    def run(self):
        # self.val_epoch()
        if self.mode == 'test':
            state_dict = torch.load(os.path.join(self.output_dir, 'best_recall_model.pth'))
            state_dict = self.convert_state_dict(state_dict)
            self.model.load_state_dict(state_dict)
            self.val_epoch()
            return
        while self.epoch <= self.epochs:
            self.train_epoch()
            self.val_epoch()
            self.epoch += 1
            self.save_checkpoint()

    def train_epoch(self):
        self.training()

        epoch_loss = 0.
        p_bar = tqdm(self.train_dataloader, desc=f'<{self.exp_name}> Train Epoch[{self.epoch}/{self.epochs}]')
        for image, label in p_bar:
            image = image.to(self.device)
            label = label.to(self.device)
            logits = self.model(image)
            loss = F.cross_entropy(logits, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            p_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss /= len(self.train_dataloader)
        print(f'Train Epoch: [{self.epoch}/{self.epochs}] Average loss: {epoch_loss:.4f}')
        self.writer.add_scalar('train/loss', epoch_loss, self.epoch)

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        label_list = []
        pred_list = []
        epoch_loss = 0.

        p_bar = tqdm(self.val_dataloader, desc=f'<{self.exp_name}> Val Epoch[{self.epoch}/{self.epochs}]')
        for image, label in p_bar:
            image = image.to(self.device)
            label = label.to(self.device)
            label_list.extend(label.cpu().numpy())

            logits = self.model(image)
            loss = F.cross_entropy(logits, label)
            epoch_loss += loss.item()
            pred = self.model.predict(image)
            pred_list.append(pred)

        epoch_loss /= len(self.val_dataloader)
        accuracy, precision, recall = accuracy_precision_recall2(pred_list, label_list)
        print(f'Val Epoch: [{self.epoch}/{self.epochs}] Loss: {epoch_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

        if self.mode == 'train':
            self.writer.add_scalar('val/loss', epoch_loss, self.epoch)
            self.writer.add_scalar('val/accuracy', accuracy, self.epoch)
            self.writer.add_scalar('val/precision', precision, self.epoch)
            self.writer.add_scalar('val/recall', recall, self.epoch)

            if accuracy > self.accuracy:
                self.save_model('best_accuracy_model.pth')
                self.accuracy = accuracy
            if precision > self.precision:
                self.save_model('best_precision_model.pth')
                self.precision = precision
            if recall > self.recall:
                self.save_model('best_recall_model.pth')
                self.recall = recall
            if epoch_loss < self.loss:
                self.save_model('best_loss_model.pth')
                self.loss = epoch_loss

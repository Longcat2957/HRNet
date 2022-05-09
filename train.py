import os
import torch
import time
import copy


from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import PoseHRNet
# from utils


current_path = Path(os.path.dirname(os.path.abspath(__file__)))  # ../HRNET/utils 의 절대 경로
weight_path = current_path / 'weight'                            # ../HRNet/data/mpii의 절대 경로

def train(model:nn.Module, criterion, optimizer, scheduler, num_epochs:int=25):
    """
    Model train function
    =====================
    
    Attributes:
    ------------
    model:nn.Module # 모델 선언
    criterion # loss function 선언
    optimizer # 최적화 함수 선언
    scheduler # lr 스케쥴러
    num_epochs # 전체 이포크 (int)
    """
    
    since = time.time() # 시작 시간을 표시
    
    # hardware config
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 초기값 설정 부분
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Dataset, DataLoader Configure
    dataset = Dataset()
    dataloader = DataLoader()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        # 각 이포크는 학습 단계와 검증 단계를 순서대로 수행합니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                
            else: # phase == 'eval'
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # 데이터를 반복한다.
            for inputs, labels in dataloader[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 매개변수 경사도를 0으로 설정한다
                optimizer.zero_grad()
                
                # forward propagation
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)           # 순전파에 대한 예측을 실행
                    _, preds = torch.max(outputs, 1)  # 가장 높은 확를을 가진 클래스를 리턴
                    loss = criterion(outputs, labels) # 손실함수에 의한 Loss 계산
                    
                    if phase == 'train':
                        loss.backward()               # back propagation
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset[phase].__len__()
            epoch_acc = running_corrects.double() / dataset[phase].__len__()
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                file_path = weight_path / '{since}_{epoch}.pth'
                torch.save(best_model_wts, file_path)
                
                
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model
    

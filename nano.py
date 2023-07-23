import torch
import torch.nn as nn
from torch.nn import functional as F
import subprocess
import string

torch.manual_seed(777)

batch_size = 32
block_size = 8
max_iters = 3000 #'더 많은 학습'을 시도하는 방법 중 하나로 max_iters 변수의 값을 더 크게 설정
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# 데이터셋 다운로드하기
#wget_command = "wget https://raw.githubusercontent.com/shop2world/data/master/input.txt"
#subprocess.run(wget_command, shell=True)

import subprocess

# 데이터셋 다운로드하기 (curl 사용)
curl_command = "curl -O https://raw.githubusercontent.com/shop2world/data/master/input.txt"
subprocess.run(curl_command, shell=True)

# 데이터셋을 읽어서 확인하기
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 문자열을 정수 리스트로 변환하고, 그 반대로 정수 리스트를 문자열로 변환하는 과정
chars = sorted(list(set(text + string.ascii_letters + string.digits + string.punctuation)))
print(len(chars))
# 인코딩 및 디코딩 함수 정의
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])
#트레인 , test 분리
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
'''
get_batch 함수: 학습 데이터를 배치 단위로 가져오는 함수입니다.
train_data와 val_data에서 랜덤하게 배치 데이터를 추출하여 입력(x)과 대상(y)으로 분할합니다.
'''
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
# 언어 모델을 정의하는 함수
class LanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

vocab_size = len(chars)
m = LanguageModel(vocab_size)
m = m.to(device)
# 모델 초기화와 옵티마이저 설정을 함수로 정의
def initialize_model_and_optimizer():
    model = LanguageModel(vocab_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer

model, optimizer = initialize_model_and_optimizer()
# 학습과 검증 데이터의 손실을 추정하는 함수 정의
'''
*새로 추가한 부분
5.estimate_loss 함수: 모델의 평가 손실을 추정하는 함수입니다. 
주어진 반복 횟수(eval_iters)만큼 학습 데이터와 검증 데이터에 대해 손실을 계산하여 평균을 구합니다.
!학습과 검증 데이터의 손실을 추정하는 함수 estimate_loss를 정의하였습니다. 
함수 내부에서는 입력으로 받은 모델을 평가 모드로 변경하고(model.eval()), 주어진 횟수(eval_iters)만큼 
반복하여 손실을 계산합니다. 계산된 손실들의 평균을 구하여 학습과 검증 데이터의 손실을 추정합니다. 
이후 모델을 다시 학습 모드로 변경합니다(model.train()). 
이렇게 함수로 정의하면 코드를 더 간결하고 모듈화된 형태로 관리할 수 있습니다. 
함수를 호출하여 학습과 검증 데이터의 손실을 추정하고 출력합니다.
'''
def estimate_loss(model, get_batch, eval_iters):
    model.eval()
    losses_dict = {}
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            losses_dict[split] = losses.mean()
    model.train()
    return losses_dict

# 함수 호출하여 손실 추정
losses_dict = estimate_loss(model, get_batch, eval_iters)
print(f"step {max_iters}: train loss {losses_dict['train']:.4f}, val loss {losses_dict['val']:.4f}")
# 학습과 생성을 수행하는 함수 정의
def train_and_generate(model, optimizer, max_iters, eval_interval, estimate_loss, get_batch):
    for iter in range(max_iters):
        # 일정 간격으로 train과 val 데이터에 대한 손실을 평가합니다
        if iter % eval_interval == 0:
            losses = estimate_loss(model, get_batch, eval_iters)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 데이터의 배치를 무작위로 추출합니다
        xb, yb = get_batch('train')

        # 손실을 평가합니다
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 모델로부터 생성합니다
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# 함수 호출
train_and_generate(model, optimizer, max_iters, eval_interval, estimate_loss, get_batch)

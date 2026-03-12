"""
Student Submission Template - Implement Your RNN Agent
"""

import torch
import torch.nn as nn
import numpy as np
import os

# ========== Fill in Your Information ==========
STUDENT_INFO = {
    "name": "AI Assistant",
    "student_id": "12345678",
    "team_name": "TronWinner",
    "description": "2-Layer GRU Hidden 96"
}


# ========== Define Your Model ==========
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 GRU 代替 LSTM
        # input_size=10
        # hidden_size=96 (这个数值经过计算，刚好卡在 100k 限制内)
        # num_layers=2 (双层网络，比单层更聪明)
        # dropout=0.1 (防止过拟合)
        self.gru = nn.GRU(
            input_size=10, 
            hidden_size=96, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.1
        )
        
        # 全连接层
        self.fc = nn.Linear(96, 4)
    
    def forward(self, x, hidden=None):
        # GRU 的 hidden 只是一个 tensor，不像 LSTM 是 (h, c) 的元组
        # 这让代码更简单，也节省内存
        out, hidden = self.gru(x, hidden)
        
        # 取序列最后一个时间步的输出
        last_output = out[:, -1, :]
        
        return self.fc(last_output), hidden


# ========== Agent Class (Do Not Modify Class Name) ==========\
class StudentAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = STUDENT_INFO.get("team_name") or STUDENT_INFO["name"]
        self.info = STUDENT_INFO
        self.model = MyModel()
        self.hidden = None # 用于在游戏中保存 LSTM 的记忆
    
    def reset(self):
        """每局游戏开始前重置记忆"""
        self.hidden = None
    
    def get_action(self, obs):
        """
        根据当前观察值 obs (numpy array, shape=(10,)) 决定动作
        """
        # 1. 预处理：将 numpy 转为 tensor
        # LSTM 需要 3D 输入: (Batch=1, Seq=1, Input=10)
        x = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad(): # 推理模式，不需要计算梯度
            # 2. 模型前向传播
            # 注意：我们将 self.hidden 传入，并将新的 self.hidden 保存下来
            # 这样 LSTM 就能记得"之前发生了什么"
            out, self.hidden = self.model(x, self.hidden)
            
            # 3. 选择分数最高的动作 (Argmax)
            action = torch.argmax(out, dim=1).item()
            
        return action

# ========== Main Program (Training Loop) ==========
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    if args.train:
        # 这里假设你目录下有 train_X.npy 和 train_Y.npy
        # 如果没有，你需要先运行相关代码生成或下载数据
        if not os.path.exists("train_X.npy") or not os.path.exists("train_Y.npy"):
            print("Error: train_X.npy or train_Y.npy not found!")
            print("Please make sure data files are in the same directory.")
        else:
            print("Loading data...")
            X = np.load("train_X.npy") # Shape通常是 (N_Samples, Seq_Len, 10)
            Y = np.load("train_Y.npy") # Shape通常是 (N_Samples,)
            
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X), torch.LongTensor(Y)
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            agent = StudentAgent()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
            
            print(f"Starting training for {args.epochs} epochs...")
            for epoch in range(args.epochs):
                total_loss = 0
                correct = 0
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    # 训练时通常不使用 hidden state 的连续传递 (stateless training on sequences)
                    # 或者数据本身就是切好的片段
                    out, _ = agent.model(batch_x) 
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    correct += (torch.argmax(out, dim=1) == batch_y).sum().item()
                
                acc = 100 * correct / len(X)
                print(f"Epoch {epoch+1}/{args.epochs}: Loss={total_loss/len(loader):.4f}, Acc={acc:.2f}%")
            
            # Save model
            os.makedirs("submissions", exist_ok=True)
            save_path = f"submissions/{STUDENT_INFO['name'].replace(' ', '_').lower()}_agent.pth"
            torch.save(agent.state_dict(), save_path)
            print(f"\nSaved to {save_path}")
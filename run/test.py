import numpy as np
import torch
from dataset.SEEDVIG import SEEDVIG
from model.stage_2 import Stage2
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

def fit(
        func_area: list = None,
        batch_size: int = 128,
        depth: int = 4,
        encoder_dim: int = 16,
        num_heads: int = 8,
        num_classes: int = 3,
        aggregation_type: str = None,
):
    random_list = np.load("./random_list.npy").tolist()
    regions = len(func_area)
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []
    for sub in range(23):
        test_dataset = SEEDVIG(dataset_name="test", normalize="minmax", subject_idx=sub, rand_list=random_list,
                               func_areas=func_area)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        attn_mask=test_dataset.attn_mask
        pe_coordination=test_dataset.coordination

        stage2 = Stage2(channel_num=17 + regions, attn_mask=attn_mask, pe_coordination=pe_coordination,
                            encoder_dim=encoder_dim, regions=regions,
                            num_heads=num_heads, depth=depth, num_class=num_classes, func_area=func_area,
                            aggregation_type=aggregation_type).to(device)

        stage2_dict = torch.load(f"../Checkpoint/dict_{sub}.pth")
        stage2.load_state_dict(stage2_dict)
        loss_fn = nn.CrossEntropyLoss()
        loss_res=0
        y_true, y_pred = [], []
        stage2.eval()
        for d, l in test_loader:

            d = d.to(device)
            l = l.to(device).long()
            output = stage2(d).to(torch.float)

            preds_class = torch.argmax(output, dim=1)
            y_true.extend(l.cpu().numpy())
            y_pred.extend(preds_class.cpu().numpy())

            loss = loss_fn(output, l)
            loss_res += loss

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        metrics = (acc, prec, rec, f1)

        print(f"subject:{sub}   best_acc:{metrics[0]}  prec:{metrics[1]}  rec:{metrics[2]}  f1:{metrics[3]}  total_loss:{loss_res}")

        all_acc.append(metrics[0])
        all_prec.append(metrics[1])
        all_rec.append(metrics[2])
        all_f1.append(metrics[3])

    avg_acc = np.mean(all_acc)
    avg_prec = np.mean(all_prec)
    avg_rec = np.mean(all_rec)
    avg_f1 = np.mean(all_f1)

    print(f"acc={avg_acc}, prec={avg_prec}, rec={avg_rec}, f1={avg_f1}")

if __name__ == "__main__":

    func_area = [[0, 2, 4], [1, 3, 5], [6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]
    fit(func_area=func_area, batch_size=128, depth=4, encoder_dim=16, num_heads=8,aggregation_type="prototype-attention")



import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import os 
import numpy as np
from sklearn.metrics import average_precision_score
from dice_loss import dice_coeff

def eval_net(args,net,loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    with tqdm(total=n_val, desc='Validation' if not args.test else 'Testing', unit='img', leave=False) as pbar:
        for batch in loader:
            img, mask, name = batch['image'], batch['mask'],batch['name']
            img = img.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                mask_pred = net(img)
                mask_save = torch.sigmoid(mask_pred).squeeze(0).squeeze(0).cpu().detach().numpy()
                #
                if not args.test:
                    Image.fromarray(mask_save/np.max(mask_save) *255)\
                        .convert('RGB').save(os.path.join('./records/valid/segmentation',name[0]+'.png'))                
                    pred = torch.sigmoid(mask_pred)
                    pred = (pred > 0.1).float()
                    tot += dice_coeff(pred, mask, args).item()
                else:
                    Image.fromarray(mask_save /np.max(mask_save)*255)\
                        .convert('RGB').save(os.path.join('./records/test/segmentation',name[0]+'.png'))
            pbar.update()
    return tot / n_val

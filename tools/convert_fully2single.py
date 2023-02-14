import torch

latest = torch.load("***/votenet_scannet_fully.pth",
                    map_location=torch.device('cpu'))  # the best training pth
for k, v in list(latest['state_dict'].items()):
    if 't_backbone' in k or 'label_encoder' in k or 'anno' in k or 'attention' in k:
        latest["state_dict"].pop(k)
for k, v in list(latest['state_dict'].items()):
    if 'backbone' in k:
        tmp = k
        tmp_after = k.replace('s_backbone.', 'backbone.')
        latest['state_dict'][tmp_after] = latest['state_dict'][k]
        latest["state_dict"].pop(k)
torch.save(latest, '***/votenet_scannet_final.pth')

print("convert finish!")

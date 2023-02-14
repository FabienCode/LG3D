_base_ = [
    '../_base_/datasets/sunrgbd-3d-10class.py', '../_base_/models/lg3d_votenet',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    anno_descriptor=dict(
        type='pointnet',
        input_channel=19,  # sunrgbd

    ),
    bbox_head=dict(
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[
                [2.114256, 1.620300, 0.927272], [0.791118, 1.279516, 0.718182],
                [0.923508, 1.867419, 0.845495], [0.591958, 0.552978, 0.827272],
                [0.699104, 0.454178, 0.75625], [0.69519, 1.346299, 0.736364],
                [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424],
                [0.404671, 1.071108, 1.688889], [0.76584, 1.398258, 0.472728]
            ]),
    ))

# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 0.004  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[48, 64])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=72)


# configs/sparse_rcnn_r50_fpn_voc.py

_base_ = [
    '_base_/models/sparse_rcnn_r50_fpn.py',
    '_base_/schedules/schedule_1x.py', 
    '_base_/default_runtime.py'
]

# 数据集配置
dataset_type = 'CocoDataset'
data_root = 'data/'

# VOC类别
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
          'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
          'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


# 数据加载配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 数据集配置
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'voc2007_trainval.json',
        img_prefix=data_root + 'VOCdevkit/VOC2007/JPEGImages/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'voc2007_test.json',
        img_prefix=data_root + 'VOCdevkit/VOC2007/JPEGImages/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'voc2007_test.json',
        img_prefix=data_root + 'VOCdevkit/VOC2007/JPEGImages/',
        classes=classes,
        pipeline=test_pipeline))

# 评估配置
evaluation = dict(
    interval=1,
    metric='bbox',
    save_best='bbox_mAP'
)

# 优化器配置
optimizer = dict(
    type='AdamW',
    lr=0.000025,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# 学习率调度
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[27, 33])

# 运行时配置
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)

# 日志配置
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

# 工作目录
work_dir = './work_dirs/sparse_rcnn'

# 预训练模型
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth'

# GPU配置
gpu_ids = range(1)

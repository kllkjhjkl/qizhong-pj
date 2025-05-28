_base_ = [
    'configs/_base_/models/mask_rcnn_r50_fpn.py',
    'configs/_base_/schedules/schedule_1x.py', 
    'configs/_base_/default_runtime.py',

]


# 修正模型类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)
    )
)

dataset_type = 'CocoDataset'
data_root = 'data/voc_coco_format2/'
data_root2 = 'data/VOCdevkit/VOC2012/'
# VOC 2012 的20个类别
classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# 图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

# 训练数据处理流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 800)],  # 多尺度训练
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

# 测试数据处理流水线
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

# 数据加载器配置
data = dict(
    samples_per_gpu=2,  # 根据GPU内存调整
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root2 + 'JPEGImages/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root2 + 'JPEGImages/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root2 + 'JPEGImages/',
        classes=classes,
        pipeline=test_pipeline))

# 评估配置
evaluation = dict(
    interval=1,  # 每个epoch评估一次
    metric=['bbox', 'segm'],  # 同时评估检测和分割
    save_best='segm_mAP',  # 保存分割mAP最好的模型
    classwise=True)  # 显示每个类别的结果

# 工作目录
work_dir = './work_dirs/mask_rcnn_r50_fpn_voc2007'
# ====== TensorBoard 可视化配置 ======
# 配置日志记录
log_config = dict(
    interval=50,  # 每50个iteration记录一次训练loss
    hooks=[
        dict(type='TextLoggerHook'),
        # TensorBoard日志钩子 - MMDetection内置支持
        dict(
            type='TensorboardLoggerHook',
            log_dir=work_dir + '/tensorboard_logs',  # TensorBoard日志保存目录
            interval=50,  # 训练loss记录间隔
            ignore_last=False,
            reset_flag=False,
            by_epoch=False  # 按iteration记录，而不是epoch
        )
    ]
)

# 检查点保存配置
checkpoint_config = dict(
    interval=1,  # 每个epoch保存一次检查点
    max_keep_ckpts=3,  # 最多保留3个检查点
    save_optimizer=True,
    save_last=True,
    create_symlink=False
)
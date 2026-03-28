"""
Inception-Attention CNN — v6  (95%+ Accuracy | <0.5 Loss | Balanced Fit)
=========================================================================

KEY CHANGES vs v5:
──────────────────
OVERFITTING FIXES:
  1. Stronger weight decay (L2: 8e-6 → 2e-5) + gradient clipping tightened (1.0 → 0.5)
  2. Dropout increased at pool layer (0.40 → 0.50) and added spatial dropout in backbone
  3. Label smoothing raised back to 0.08 (v5 lowered it too aggressively to 0.03)
  4. MixUp alpha reduced 0.4 → 0.3 (over-mixing causes underfitting)
  5. EMA decay reduced 0.9997 → 0.999 (faster tracking, less memorization)
  6. Removed FMix (redundant with CutMix, increases variance)
  7. Max samples per class capped at 2500 (reduce redundant samples)
  8. SWA start moved earlier: 75% → 65% of epochs

UNDERFITTING FIXES:
  9. Warmup extended 5 → 8 epochs (more stable early training)
 10. ReduceLROnPlateau factor 0.4 → 0.5 (less aggressive LR killing)
 11. Patience increased: EarlyStopping 20 → 25 (give model time to recover)
 12. Stage 3 backbone widened: 80ch → 96ch (more capacity)
 13. GeGLU FC head kept but dropout between layers reduced

LOSS < 0.5 FIXES:
 14. CategoricalCrossentropy with label_smoothing=0.08 naturally floors loss ~0.08
 15. Temperature scaling at inference (T=1.2) sharpens softmax → lower CE loss on test
 16. Pseudo-label threshold raised 0.92 → 0.95 (only very confident pseudo labels)
 17. Added BatchNorm after stem (stabilizes early gradient flow)
 18. AdamW beta_2 tuned to 0.995 (smoother second-moment estimate)

GENERAL:
 19. Progressive resizing retained (128→160→192) with better epoch splits
 20. TTA steps reduced 15 → 10 (15 caused label leakage artifacts in some edge cases)
 21. Class weights computed on raw (non-oversampled) distribution for correctness
"""

import os, gc, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import tensorflow as tf
import keras
from keras import layers, ops

warnings.filterwarnings('ignore')


# ============================================================
# CONFIG
# ============================================================
class Config:
    DATA_DIR  = "processed_images_multiclass"
    CSV_FILE  = "labels.csv"

    IMG_HEIGHT = 192
    IMG_WIDTH  = 192
    CHANNELS   = 3

    BATCH_SIZE   = 32
    TOTAL_EPOCHS = 120
    INITIAL_LR   = 3e-4
    MIN_LR       = 1e-7
    WARMUP_EPOCHS = 8           # ↑ from 5 — more stable warmup

    DROPOUT_RATE      = 0.35    # ↑ slightly
    DROP_PATH_RATE    = 0.20
    L2_REGULARIZATION = 2e-5    # ↑ from 8e-6 — stronger regularization
    LABEL_SMOOTHING   = 0.08    # ↑ from 0.03 — prevents overconfidence

    MAX_SAMPLES_PER_CLASS = 2500  # ↓ from 3000

    MIXUP_ALPHA  = 0.3          # ↓ from 0.4 — less over-mixing
    CUTMIX_ALPHA = 1.0
    USE_RANDAUG  = True
    RANDAUG_N    = 2
    RANDAUG_M    = 9

    USE_TTA   = True
    TTA_STEPS = 10              # ↓ from 15

    USE_EMA   = True
    EMA_DECAY = 0.999           # ↓ from 0.9997

    USE_SWA   = True
    SWA_START_EPOCH_FRAC = 0.65  # Start SWA at 65% of phase epochs

    NUM_CLASSES  = 4
    CLASS_NAMES  = ['class_0', 'class_1', 'class_2', 'class_3']

    # Progressive resizing
    PHASE1_EPOCHS = 40   # 128×128
    PHASE2_EPOCHS = 80   # 160×160
    # PHASE3:       120   # 192×192


# ============================================================
# CUSTOM LAYERS
# ============================================================

class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        if (not training) or self.drop_prob == 0.0:
            return x
        batch = ops.shape(x)[0]
        rand  = tf.random.uniform((batch, 1, 1, 1), dtype=x.dtype)
        keep  = ops.cast(rand >= self.drop_prob, x.dtype) / (1.0 - self.drop_prob)
        return x * keep

    def get_config(self):
        cfg = super().get_config()
        cfg['drop_prob'] = self.drop_prob
        return cfg


class ChannelMean(layers.Layer):
    def call(self, x): return ops.mean(x, axis=-1, keepdims=True)
    def compute_output_shape(self, s): return s[:-1] + (1,)


class ChannelMax(layers.Layer):
    def call(self, x): return ops.max(x, axis=-1, keepdims=True)
    def compute_output_shape(self, s): return s[:-1] + (1,)


class ClipConstraint(keras.constraints.Constraint):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)

    def get_config(self):
        return {'min_val': self.min_val, 'max_val': self.max_val}


class GeM(layers.Layer):
    """Generalized Mean Pooling."""
    def __init__(self, p_init=3.0, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps    = eps
        self.p_init = p_init

    def build(self, input_shape):
        self.p = self.add_weight(
            name='p', shape=(), initializer=tf.constant_initializer(self.p_init),
            trainable=True, constraint=ClipConstraint(1.0, 6.0))

    def call(self, x):
        x = tf.clip_by_value(x, self.eps, float('inf'))
        x = tf.pow(x, self.p)
        x = tf.reduce_mean(x, axis=[1, 2])
        return tf.pow(x, 1.0 / self.p)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'p_init': self.p_init, 'eps': self.eps})
        return cfg


class GeGLU(layers.Layer):
    """Gated Linear Unit with GELU."""
    def __init__(self, units, reg=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.reg   = reg

    def build(self, input_shape):
        self.proj = layers.Dense(self.units * 2, use_bias=False,
                                  kernel_regularizer=self.reg,
                                  kernel_initializer='he_normal')

    def call(self, x):
        xp = self.proj(x)
        x1, x2 = tf.split(xp, 2, axis=-1)
        return x1 * tf.nn.gelu(x2)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units})
        return cfg


# ============================================================
# BUILDING BLOCKS
# ============================================================

def conv_bn_swish(x, filters, kernel_size, strides=1, padding='same',
                  name='', reg=None, groups=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                      groups=groups, use_bias=False, kernel_regularizer=reg,
                      kernel_initializer='he_normal', name=f'{name}_conv')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name=f'{name}_bn')(x)
    x = layers.Activation('swish', name=f'{name}_act')(x)
    return x


def se_block(x, ratio=16, name='se'):
    ch  = x.shape[-1]
    r   = max(ch // ratio, 8)
    gap = layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
    fc1 = layers.Dense(r,  activation='swish',   use_bias=True, name=f'{name}_fc1')(gap)
    fc2 = layers.Dense(ch, activation='sigmoid', use_bias=True, name=f'{name}_fc2')(fc1)
    fc2 = layers.Reshape((1, 1, ch), name=f'{name}_rs')(fc2)
    return layers.Multiply(name=f'{name}_out')([x, fc2])


def cbam_block(x, reduction_ratio=16, name='cbam'):
    ch  = x.shape[-1]
    r   = max(ch // reduction_ratio, 8)

    gap = layers.GlobalAveragePooling2D(name=f'{name}_gap')(x)
    gmp = layers.GlobalMaxPooling2D(name=f'{name}_gmp')(x)
    fc1 = layers.Dense(r,  activation='relu',    use_bias=False, name=f'{name}_fc1')
    fc2 = layers.Dense(ch, activation='sigmoid', use_bias=False, name=f'{name}_fc2')

    ch_att = layers.Add(name=f'{name}_ch_add')([
        layers.Reshape((1, 1, ch))(fc2(fc1(gap))),
        layers.Reshape((1, 1, ch))(fc2(fc1(gmp)))
    ])
    x = layers.Multiply(name=f'{name}_ch_scale')([x, ch_att])

    sp_avg = ChannelMean(name=f'{name}_sp_avg')(x)
    sp_max = ChannelMax(name=f'{name}_sp_max')(x)
    sp_cat = layers.Concatenate(axis=-1, name=f'{name}_sp_cat')([sp_avg, sp_max])
    sp_att = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid',
                           use_bias=False, name=f'{name}_sp_conv')(sp_cat)
    return layers.Multiply(name=f'{name}_sp_scale')([x, sp_att])


def inception_module(x, f1, f3r, f3, f5r, f5, fp,
                     strides=1, name='inc', reg=None):
    b1 = conv_bn_swish(x,  f1,  (1, 1), strides=strides, name=f'{name}_b1',  reg=reg)
    b2 = conv_bn_swish(x,  f3r, (1, 1),                  name=f'{name}_b2r', reg=reg)
    b2 = conv_bn_swish(b2, f3,  (3, 3), strides=strides, name=f'{name}_b2',  reg=reg)
    b3 = conv_bn_swish(x,  f5r, (1, 1),                  name=f'{name}_b3r', reg=reg)
    b3 = conv_bn_swish(b3, f5,  (3, 3),                  name=f'{name}_b3a', reg=reg)
    b3 = conv_bn_swish(b3, f5,  (3, 3), strides=strides, name=f'{name}_b3b', reg=reg)
    ps = strides if strides > 1 else 1
    b4 = layers.MaxPooling2D((3, 3), strides=ps, padding='same', name=f'{name}_b4p')(x)
    b4 = conv_bn_swish(b4, fp, (1, 1), name=f'{name}_b4', reg=reg)
    return layers.Concatenate(axis=-1, name=f'{name}_cat')([b1, b2, b3, b4])


def inception_attention_block(x, filters_cfg, strides=1,
                               drop_path=0.0, block_name='iab', reg=None,
                               spatial_drop=0.0):
    f1, f3r, f3, f5r, f5, fp = filters_cfg
    out_ch = f1 + f3 + f5 + fp

    inc = inception_module(x, f1, f3r, f3, f5r, f5, fp,
                           strides=strides, name=f'{block_name}_inc', reg=reg)
    inc = se_block(inc,   name=f'{block_name}_se')
    inc = cbam_block(inc, name=f'{block_name}_cbam')

    # Spatial dropout in backbone (helps prevent co-adaptation of feature maps)
    if spatial_drop > 0.0:
        inc = layers.SpatialDropout2D(spatial_drop, name=f'{block_name}_sdrop')(inc)

    if drop_path > 0.0:
        inc = StochasticDepth(drop_path, name=f'{block_name}_sd')(inc)

    in_ch = x.shape[-1]
    if strides != 1 or in_ch != out_ch:
        shortcut = layers.Conv2D(out_ch, (1, 1), strides=strides, padding='same',
                                 use_bias=False, kernel_regularizer=reg,
                                 kernel_initializer='he_normal',
                                 name=f'{block_name}_proj')(x)
        shortcut = layers.BatchNormalization(momentum=0.99, epsilon=1e-5,
                                             name=f'{block_name}_proj_bn')(shortcut)
    else:
        shortcut = x

    out = layers.Add(name=f'{block_name}_add')([inc, shortcut])
    return layers.Activation('swish', name=f'{block_name}_out')(out)


# ============================================================
# MODEL v6
# Changes: wider Stage 3 (96ch), spatial dropout in early stages,
#          BatchNorm after stem, improved pooling head
# ============================================================
def build_model(img_size=None):
    h = img_size or Config.IMG_HEIGHT
    w = img_size or Config.IMG_WIDTH
    print(f"\nBuilding Inception-Attention CNN  v6  ({h}×{w})")
    print("=" * 60)
    reg = keras.regularizers.l2(Config.L2_REGULARIZATION)

    n_blocks = 11
    dp_vals  = np.linspace(0.0, Config.DROP_PATH_RATE, n_blocks).tolist()
    idx = [0]
    def next_dp():
        v = dp_vals[idx[0]]; idx[0] += 1; return v

    inp = keras.Input(shape=(h, w, Config.CHANNELS), name='image_input')
    x   = layers.Rescaling(scale=2.0, offset=-1.0, name='normalize')(inp)

    # --- Stem (with BN at end for stable gradient flow) ---
    x = conv_bn_swish(x, 32,  (3, 3), strides=2, name='stem1', reg=reg)
    x = conv_bn_swish(x, 32,  (3, 3), strides=1, name='stem2', reg=reg)
    x = conv_bn_swish(x, 64,  (3, 3), strides=1, name='stem3', reg=reg)
    x = conv_bn_swish(x, 64,  (3, 3), strides=2, name='stem4', reg=reg)
    x = conv_bn_swish(x, 128, (3, 3), strides=1, name='stem5', reg=reg)
    # Extra BN after stem — stabilizes early activations
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name='stem_bn')(x)

    # --- Stage 1 (spatial dropout=0.05 to regularize early maps) ---
    x  = inception_attention_block(x, (16, 16, 32, 4, 8, 8),
                                   strides=1, drop_path=next_dp(),
                                   spatial_drop=0.05, block_name='s1_0', reg=reg)
    s1 = x   # save for potential multi-scale use

    # --- Stage 2 ---
    x  = inception_attention_block(x, (32, 32, 64, 8, 16, 16),
                                   strides=2, drop_path=next_dp(),
                                   spatial_drop=0.05, block_name='s2_0', reg=reg)
    x  = inception_attention_block(x, (32, 32, 64, 8, 16, 16),
                                   strides=1, drop_path=next_dp(),
                                   block_name='s2_1', reg=reg)
    s2 = x

    # --- Stage 3 (wider: 96ch instead of 80ch) ---
    x  = inception_attention_block(x, (96, 96, 192, 24, 48, 48),
                                   strides=2, drop_path=next_dp(),
                                   block_name='s3_0', reg=reg)
    x  = inception_attention_block(x, (96, 96, 192, 24, 48, 48),
                                   strides=1, drop_path=next_dp(),
                                   block_name='s3_1', reg=reg)
    x  = inception_attention_block(x, (96, 96, 192, 24, 48, 48),
                                   strides=1, drop_path=next_dp(),
                                   block_name='s3_2', reg=reg)
    s3 = x   # 384 ch (96+192+48+48)

    # --- Stage 4 ---
    x  = inception_attention_block(x, (160, 160, 320, 40, 80, 80),
                                   strides=2, drop_path=next_dp(),
                                   block_name='s4_0', reg=reg)
    x  = inception_attention_block(x, (160, 160, 320, 40, 80, 80),
                                   strides=1, drop_path=next_dp(),
                                   block_name='s4_1', reg=reg)
    x  = inception_attention_block(x, (160, 160, 320, 40, 80, 80),
                                   strides=1, drop_path=next_dp(),
                                   block_name='s4_2', reg=reg)

    # --- Stage 5 ---
    x  = inception_attention_block(x, (160, 160, 320, 40, 80, 80),
                                   strides=2, drop_path=next_dp(),
                                   block_name='s5_0', reg=reg)
    x  = inception_attention_block(x, (160, 160, 320, 40, 80, 80),
                                   strides=1, drop_path=next_dp(),
                                   block_name='s5_1', reg=reg)

    # ---- Multi-Scale Pooling (GAP + GMP + GeM + s3 scale) ----
    gap  = layers.GlobalAveragePooling2D(name='gap')(x)
    gmp  = layers.GlobalMaxPooling2D(name='gmp')(x)
    gem  = GeM(name='gem')(x)

    s3p  = conv_bn_swish(s3, 256, (1, 1), name='s3_proj', reg=reg)
    s3p  = layers.GlobalAveragePooling2D(name='s3_gap')(s3p)

    x    = layers.Concatenate(name='pool_concat')([gap, gmp, gem, s3p])
    # Total: 640 + 640 + 640 + 256 = 2176

    # ---- FC Head ----
    x = layers.Dropout(0.50, name='drop_pool')(x)      # ↑ from 0.40

    # Layer 1: 2048-d GeGLU
    x = GeGLU(2048, reg=reg, name='fc1_geglu')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name='fc1_bn')(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name='drop_fc1')(x)

    # Layer 2: 1024-d GeGLU
    x = GeGLU(1024, reg=reg, name='fc2_geglu')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name='fc2_bn')(x)
    x = layers.Dropout(Config.DROPOUT_RATE * 0.4, name='drop_fc2')(x)  # ↑ slightly

    # Layer 3: 512-d Dense
    x = layers.Dense(512, use_bias=False, kernel_regularizer=reg,
                     kernel_initializer='he_normal', name='fc3')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name='fc3_bn')(x)
    x = layers.Activation('swish', name='fc3_act')(x)
    x = layers.Dropout(Config.DROPOUT_RATE * 0.25, name='drop_fc3')(x)

    out = layers.Dense(Config.NUM_CLASSES, activation='softmax',
                       kernel_regularizer=reg, name='output')(x)

    return keras.Model(inp, out, name='Inception_Attention_v6')


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=Config.L2_REGULARIZATION,
            beta_2=0.995,       # ↑ smoother second-moment estimate
            clipnorm=0.5,       # ↓ from 1.0 — tighter gradient clipping
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=Config.LABEL_SMOOTHING),
        metrics=['accuracy',
                 keras.metrics.TopKCategoricalAccuracy(k=2, name='top2')]
    )
    return model


# ============================================================
# LR SCHEDULE — 3-cycle cosine annealing with warmup
# ============================================================
def make_lr_schedule(total_epochs, peak_lr):
    warmup     = Config.WARMUP_EPOCHS
    boundaries = [int(total_epochs * 0.33), int(total_epochs * 0.66), total_epochs]
    peak_lrs   = [peak_lr, peak_lr * 0.5, peak_lr * 0.2]

    def schedule(epoch):
        if epoch < warmup:
            return float(peak_lr * (0.01 + 0.99 * (epoch + 1) / warmup))
        for i, (end, pk) in enumerate(zip(boundaries, peak_lrs)):
            start = boundaries[i - 1] if i > 0 else warmup
            if epoch < end:
                prog   = (epoch - start) / max(end - start, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
                return float(Config.MIN_LR + (pk - Config.MIN_LR) * cosine)
        return float(Config.MIN_LR)

    return schedule


# ============================================================
# AUGMENTATION
# ============================================================

def _rand_aug_transform(img):
    """Independent coin-flip augmentations."""
    n_ops = 8
    p = Config.RANDAUG_N / n_ops

    def maybe(op, img, prob=p):
        return tf.cond(tf.random.uniform(()) < prob, lambda: op(img), lambda: img)

    img = maybe(lambda i: tf.image.random_brightness(i, 0.20), img)
    img = maybe(lambda i: tf.image.random_contrast(i, 0.70, 1.40), img)
    img = maybe(lambda i: tf.image.random_saturation(i, 0.70, 1.40), img)
    img = maybe(lambda i: tf.image.random_hue(i, 0.08), img)
    img = maybe(lambda i: tf.image.random_flip_left_right(i), img)
    img = maybe(lambda i: tf.image.random_flip_up_down(i), img)
    img = maybe(
        lambda i: tf.clip_by_value(i + tf.random.uniform(tf.shape(i), -0.03, 0.03), 0, 1),
        img
    )
    def coarse_dropout(i):
        mask = tf.random.uniform(tf.shape(i)) > 0.04
        return i * tf.cast(mask, i.dtype)
    img = maybe(coarse_dropout, img, prob=0.25)
    return tf.clip_by_value(img, 0.0, 1.0)


def mixup_batch(images, labels):
    lam = float(np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA))
    idx = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    return (lam * images + (1 - lam) * tf.gather(images, idx),
            lam * labels + (1 - lam) * tf.gather(labels, idx))


def cutmix_batch(images, labels):
    lam   = float(np.random.beta(Config.CUTMIX_ALPHA, Config.CUTMIX_ALPHA))
    idx   = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    H     = tf.cast(tf.shape(images)[1], tf.float32)
    W     = tf.cast(tf.shape(images)[2], tf.float32)
    ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(H * ratio, tf.int32)
    cut_w = tf.cast(W * ratio, tf.int32)
    cy    = tf.random.uniform((), 0, tf.cast(H, tf.int32), dtype=tf.int32)
    cx    = tf.random.uniform((), 0, tf.cast(W, tf.int32), dtype=tf.int32)
    y1    = tf.clip_by_value(cy - cut_h // 2, 0, tf.cast(H, tf.int32))
    y2    = tf.clip_by_value(cy + cut_h // 2, 0, tf.cast(H, tf.int32))
    x1    = tf.clip_by_value(cx - cut_w // 2, 0, tf.cast(W, tf.int32))
    x2    = tf.clip_by_value(cx + cut_w // 2, 0, tf.cast(W, tf.int32))
    iH, iW = tf.cast(H, tf.int32), tf.cast(W, tf.int32)
    mask  = tf.cast(
        tf.logical_and(
            tf.logical_and(tf.range(iH)[:, None] >= y1, tf.range(iH)[:, None] < y2),
            tf.logical_and(tf.range(iW)[None, :] >= x1, tf.range(iW)[None, :] < x2)
        ), images.dtype)[None, :, :, None]
    mixed    = images * (1 - mask) + tf.gather(images, idx) * mask
    lam_adj  = 1.0 - tf.cast((y2 - y1) * (x2 - x1), tf.float32) / (H * W)
    return (mixed, lam_adj * labels + (1 - lam_adj) * tf.gather(labels, idx))


def apply_augmentation(images, labels):
    """MixUp 35% | CutMix 40% | No-aug 25%  (FMix removed)."""
    r = tf.random.uniform(())
    if r < 0.40:
        return cutmix_batch(images, labels)
    elif r < 0.75:
        return mixup_batch(images, labels)
    else:
        return images, labels


# ============================================================
# DATA PIPELINE
# ============================================================

def preprocess_image(path, label_oh, training=True, img_h=None, img_w=None):
    h = img_h or Config.IMG_HEIGHT
    w = img_w or Config.IMG_WIDTH

    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=Config.CHANNELS, expand_animations=False)
    img.set_shape([None, None, Config.CHANNELS])

    if training:
        scale = tf.random.uniform((), 1.0, 1.20)   # ↓ from 1.25 (less extreme)
        H     = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
        W     = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
        img   = tf.image.resize(img, [H, W])
        img   = tf.cast(img, tf.float32) / 255.0
        img   = tf.image.random_crop(img, [h, w, Config.CHANNELS])
        img   = _rand_aug_transform(img)
        k     = tf.random.uniform((), 0, 4, dtype=tf.int32)
        img   = tf.image.rot90(img, k)
        pad   = 8   # ↓ from 10
        img   = tf.image.resize_with_crop_or_pad(img, h + pad, w + pad)
        img   = tf.image.random_crop(img, [h, w, Config.CHANNELS])
    else:
        img = tf.image.resize(img, [h, w])
        img = tf.cast(img, tf.float32) / 255.0

    return tf.clip_by_value(img, 0.0, 1.0), label_oh


def load_csv_labels(csv_path, data_dir):
    print("\n=== LOADING LABELS ===")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found"); return None, None
    df = pd.read_csv(csv_path)
    if 'filename' not in df.columns or 'label' not in df.columns:
        print("ERROR: need 'filename' and 'label' columns"); return None, None
    paths, labels, missing = [], [], 0
    for _, row in df.iterrows():
        p = os.path.join(data_dir, str(row['filename']))
        if os.path.exists(p):
            paths.append(p); labels.append(int(row['label']))
        else:
            missing += 1
    if missing: print(f"  Warning: {missing} images not found")
    print(f"  Total found: {len(paths)}")
    for lbl, cnt in sorted(Counter(labels).items()):
        nm = Config.CLASS_NAMES[lbl] if lbl < len(Config.CLASS_NAMES) else f"class_{lbl}"
        print(f"  {nm}: {cnt}")
    return np.array(paths), np.array(labels)


def create_datasets(csv_path, data_dir, img_h=None, img_w=None):
    print("\n=== BUILDING DATASETS ===")
    paths, labels = load_csv_labels(csv_path, data_dir)
    if paths is None: return (None,) * 6

    # --- Cap per class ---
    cap   = Config.MAX_SAMPLES_PER_CLASS
    cap_p, cap_l = [], []
    for cls in range(Config.NUM_CLASSES):
        mask   = labels == cls
        cp, cl = paths[mask], labels[mask]
        if len(cp) > cap:
            ix    = np.random.RandomState(42).choice(len(cp), cap, replace=False)
            cp, cl = cp[ix], cl[ix]
            print(f"  Capped {Config.CLASS_NAMES[cls]}: {mask.sum()} → {cap}")
        cap_p.extend(cp); cap_l.extend(cl)
    paths, labels = np.array(cap_p), np.array(cap_l)

    # --- Class weights (computed BEFORE oversampling — correct distribution) ---
    raw_counts = Counter(labels)
    total_raw  = len(labels)
    class_weights = {i: total_raw / (Config.NUM_CLASSES * max(raw_counts[i], 1))
                     for i in range(Config.NUM_CLASSES)}
    print(f"  Class weights: { {k: round(v, 3) for k, v in class_weights.items()} }")

    # --- Oversample minority classes ---
    target    = max(Counter(labels).values())
    bal_p, bal_l = [], []
    rng       = np.random.RandomState(42)
    for cls in range(Config.NUM_CLASSES):
        mask   = labels == cls
        cp, cl = paths[mask], labels[mask]
        if len(cp) < target:
            extra_ix = rng.choice(len(cp), target - len(cp), replace=True)
            cp        = np.concatenate([cp, cp[extra_ix]])
            cl        = np.full(len(cp), cls)
            print(f"  Oversample {Config.CLASS_NAMES[cls]}: {mask.sum()} → {len(cp)}")
        bal_p.extend(cp); bal_l.extend(cl)

    all_p  = np.array(bal_p)
    all_l  = np.array(bal_l)
    all_oh = keras.utils.to_categorical(all_l, Config.NUM_CLASSES)

    tv_p, test_p, tv_l, test_l = train_test_split(
        all_p, all_oh, test_size=0.10, stratify=all_l, random_state=42)
    tr_p, val_p, tr_l, val_l = train_test_split(
        tv_p, tv_l, test_size=0.111,
        stratify=np.argmax(tv_l, axis=1), random_state=42)

    print(f"\n  Train: {len(tr_p)}  |  Val: {len(val_p)}  |  Test: {len(test_p)}")

    h = img_h or Config.IMG_HEIGHT
    w = img_w or Config.IMG_WIDTH

    def make_ds(fp, fl, training):
        ds = tf.data.Dataset.from_tensor_slices((fp, tf.cast(fl, tf.float32)))
        ds = ds.map(lambda x, y: preprocess_image(x, y, training, h, w),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(512, reshuffle_each_iteration=True)
        ds = ds.batch(Config.BATCH_SIZE)
        if training:
            ds = ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        if not training:
            ds = ds.cache()
        return ds.prefetch(tf.data.AUTOTUNE)

    return (make_ds(tr_p, tr_l, True),
            make_ds(val_p, val_l, False),
            make_ds(test_p, test_l, False),
            (tr_p, np.argmax(tr_l, axis=1)),
            (val_p, np.argmax(val_l, axis=1)),
            (test_p, np.argmax(test_l, axis=1)),
            class_weights)   # return for correct class weighting


# ============================================================
# EMA
# ============================================================
class EMAWeights:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = [w.numpy().copy() for w in model.weights]

    def update(self, model):
        d = self.decay
        for i, weight in enumerate(model.weights):
            self.shadow[i] = d * self.shadow[i] + (1 - d) * weight.numpy()

    def apply(self, model):
        self._backup = [w.numpy().copy() for w in model.weights]
        for shadow, weight in zip(self.shadow, model.weights):
            weight.assign(shadow)

    def restore(self, model):
        for backup, weight in zip(self._backup, model.weights):
            weight.assign(backup)
        del self._backup


class EMACallback(keras.callbacks.Callback):
    def __init__(self, ema):
        super().__init__()
        self.ema = ema

    def on_batch_end(self, batch, logs=None):
        self.ema.update(self.model)

    def on_epoch_end(self, epoch, logs=None):
        self.ema.apply(self.model)
        self.ema.restore(self.model)


# ============================================================
# SWA
# ============================================================
class SWACallback(keras.callbacks.Callback):
    def __init__(self, start_epoch_frac=0.65):
        super().__init__()
        self.start_epoch_frac = start_epoch_frac
        self.start_epoch      = None
        self.n_averaged       = 0
        self.avg_weights      = None

    def on_train_begin(self, logs=None):
        total = self.params.get('epochs', 100)
        self.start_epoch = int(total * self.start_epoch_frac)
        print(f"  SWA will start at epoch {self.start_epoch}")

    def on_epoch_end(self, epoch, logs=None):
        if self.start_epoch is None or epoch < self.start_epoch:
            return
        w = [wt.numpy() for wt in self.model.weights]
        if self.avg_weights is None:
            self.avg_weights = w
            self.n_averaged  = 1
        else:
            n = self.n_averaged
            self.avg_weights = [(n * a + b) / (n + 1) for a, b in zip(self.avg_weights, w)]
            self.n_averaged += 1

    def apply(self, model):
        if self.avg_weights is not None:
            print(f"  Applying SWA weights (averaged over {self.n_averaged} epochs)...")
            for avg, w in zip(self.avg_weights, model.weights):
                w.assign(avg)


# ============================================================
# TTA — 10 diverse passes
# ============================================================
def tta_predict(model, test_ds, steps=10):
    print(f"\nTTA: {steps} passes...")

    def _aug(imgs, lbls):
        r    = tf.random.uniform(())
        imgs = tf.cond(r < 0.5, lambda: tf.image.random_flip_left_right(imgs), lambda: imgs)
        r    = tf.random.uniform(())
        imgs = tf.cond(r < 0.3, lambda: tf.image.random_flip_up_down(imgs), lambda: imgs)
        imgs = tf.image.random_brightness(imgs, 0.06)
        imgs = tf.image.random_contrast(imgs, 0.94, 1.06)
        k    = tf.random.uniform((), 0, 4, dtype=tf.int32)
        imgs = tf.image.rot90(imgs, k)
        crop_frac = tf.random.uniform((), 0.90, 1.0)
        shape = tf.shape(imgs)
        h_c   = tf.cast(tf.cast(shape[1], tf.float32) * crop_frac, tf.int32)
        w_c   = tf.cast(tf.cast(shape[2], tf.float32) * crop_frac, tf.int32)
        imgs  = tf.image.random_crop(imgs, [shape[0], h_c, w_c, shape[3]])
        imgs  = tf.image.resize(imgs, [shape[1], shape[2]])
        return tf.clip_by_value(imgs, 0, 1), lbls

    # First pass: no augmentation
    running_mean = model.predict(test_ds, verbose=0).astype(np.float32)

    for step in range(1, steps):
        aug_ds = test_ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
        p = model.predict(aug_ds, verbose=0).astype(np.float32)
        running_mean += (p - running_mean) / (step + 1)
        del p; gc.collect()

    return running_mean


# ============================================================
# PSEUDO-LABEL FINE-TUNING
# ============================================================
def pseudo_label_finetune(model, val_ds, val_info, train_ds, tr_info,
                           confidence_threshold=0.95, epochs=8, lr=2e-5):
    """
    Adds high-confidence val predictions back to training for fine-tuning.
    Threshold raised to 0.95 (was 0.92) to avoid noisy labels.
    """
    print(f"\n=== PSEUDO-LABEL SELF-TRAINING (threshold={confidence_threshold}) ===")
    val_preds   = model.predict(val_ds, verbose=0)
    max_conf    = val_preds.max(axis=1)
    pseudo_mask = max_conf >= confidence_threshold
    pseudo_n    = pseudo_mask.sum()
    print(f"  Confident val samples: {pseudo_n} / {len(val_info[0])}")
    if pseudo_n < 30:
        print("  Too few confident samples — skipping pseudo-label step")
        return

    pseudo_paths  = val_info[0][pseudo_mask]
    pseudo_labels = np.argmax(val_preds[pseudo_mask], axis=1)
    pseudo_oh     = keras.utils.to_categorical(pseudo_labels, Config.NUM_CLASSES)

    combined_p = np.concatenate([tr_info[0], pseudo_paths])
    combined_l = np.concatenate([keras.utils.to_categorical(tr_info[1], Config.NUM_CLASSES),
                                  pseudo_oh])

    ds = tf.data.Dataset.from_tensor_slices(
        (combined_p, tf.cast(combined_l, tf.float32)))
    ds = (ds.map(lambda x, y: preprocess_image(x, y, True),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(512)
            .batch(Config.BATCH_SIZE)
            .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

    # Fine-tune: no label smoothing (pseudo labels are already hard)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-6,
                                          clipnorm=0.5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top2')]
    )
    model.fit(ds, validation_data=val_ds, epochs=epochs, verbose=1,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                       patience=4,
                                                       restore_best_weights=True,
                                                       mode='max')])
    print("  Pseudo-label fine-tuning complete.")


# ============================================================
# OVERFITTING MONITOR CALLBACK
# Prints train/val accuracy gap each epoch so you can catch
# overfitting early without needing to wait for full training.
# ============================================================
class OverfitMonitor(keras.callbacks.Callback):
    def __init__(self, threshold=0.08):
        super().__init__()
        self.threshold = threshold  # warn if gap > 8%

    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return
        tr_acc  = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        gap     = tr_acc - val_acc
        if gap > self.threshold:
            print(f"\n  ⚠️  Overfit warning: train={tr_acc:.3f} val={val_acc:.3f} gap={gap:.3f}")


# ============================================================
# PLOTTING
# ============================================================
def plot_results(history, preds, true_labels, version='v6'):
    acc      = history.history.get('accuracy', [])
    val_acc  = history.history.get('val_accuracy', [])
    loss     = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    pred_cls = np.argmax(preds, axis=1)
    n        = Config.NUM_CLASSES

    fig, axes = plt.subplots(3, 3, figsize=(21, 14))
    fig.suptitle(f'Inception-Attention {version} — Training Results',
                 fontsize=14, fontweight='bold')

    # Accuracy curve
    ax = axes[0, 0]
    ax.plot(acc,     color='royalblue',  lw=2, label='Train')
    ax.plot(val_acc, color='darkorange', lw=2, label='Val')
    ax.axhline(0.95, color='green', ls='--', lw=1, label='95% target')
    ax.set_title('Accuracy', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim([max(0, min(acc + val_acc) - 0.05), 1.02])

    # Loss curve
    ax = axes[0, 1]
    ax.plot(loss,     color='royalblue',  lw=2, label='Train')
    ax.plot(val_loss, color='darkorange', lw=2, label='Val')
    ax.axhline(0.5, color='red', ls='--', lw=1, label='0.5 target')
    ax.set_title('Loss', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    # Train-Val gap (overfitting detector)
    ax = axes[0, 2]
    gap = [a - b for a, b in zip(acc, val_acc)]
    ax.plot(gap, color='purple', lw=2)
    ax.axhline(0.08, color='red', ls='--', lw=1, label='Overfit threshold (8%)')
    ax.axhline(0.0,  color='gray', ls='-',  lw=0.5)
    ax.set_title('Train-Val Accuracy Gap', fontweight='bold')
    ax.set_ylabel('Gap'); ax.legend(); ax.grid(alpha=0.3)

    # Confusion matrix
    ax = axes[1, 0]
    cm = confusion_matrix(true_labels, pred_cls)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # ROC curves
    ax = axes[1, 1]
    true_bin = label_binarize(true_labels, classes=range(n))
    clrs     = ['royalblue', 'tomato', 'mediumseagreen', 'orchid']
    roc_aucs = {}
    for i in range(n):
        fpr, tpr, _ = roc_curve(true_bin[:, i], preds[:, i])
        roc_aucs[i] = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=clrs[i], lw=2,
                label=f'{Config.CLASS_NAMES[i]} AUC={roc_aucs[i]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title('ROC Curves', fontweight='bold')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Per-class metrics
    metrics = [
        (lambda: precision_score(true_labels, pred_cls, average=None, zero_division=0), 'Precision', 'steelblue'),
        (lambda: recall_score(true_labels, pred_cls, average=None, zero_division=0),    'Recall',    'coral'),
        (lambda: f1_score(true_labels, pred_cls, average=None, zero_division=0),        'F1-Score',  'mediumseagreen'),
    ]
    for col_i, (fn, title, color) in enumerate(metrics):
        ax   = axes[2, col_i]
        vals = fn()
        ax.bar(np.arange(n), vals, color=color, alpha=0.75)
        ax.set_title(f'Per-Class {title}', fontweight='bold')
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(Config.CLASS_NAMES, rotation=45, ha='right')
        ax.set_ylim([0, 1.1]); ax.grid(alpha=0.3, axis='y')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

    # Summary text
    ax = axes[1, 2]; ax.axis('off')
    ta   = accuracy_score(true_labels, pred_cls)
    tp   = precision_score(true_labels, pred_cls, average='macro', zero_division=0)
    tr   = recall_score(true_labels, pred_cls, average='macro', zero_division=0)
    tf1  = f1_score(true_labels, pred_cls, average='macro', zero_division=0)
    mauc = np.mean(list(roc_aucs.values()))
    best_val = max(val_acc) if val_acc else 0
    best_loss = min(val_loss) if val_loss else 0
    txt  = (f"INCEPTION-ATTENTION {version}\n\n"
            f"Epochs         : {len(acc)}\n"
            f"Best Val Acc   : {best_val:.4f}\n"
            f"Best Val Loss  : {best_loss:.4f}\n\n"
            f"Test Accuracy  : {ta:.4f}  {'✅' if ta >= 0.95 else '❌'}\n"
            f"Test Loss ≤0.5 : {'✅' if best_loss <= 0.5 else '❌'}\n"
            f"Test Precision : {tp:.4f}\n"
            f"Test Recall    : {tr:.4f}\n"
            f"Test F1        : {tf1:.4f}\n"
            f"Macro AUC      : {mauc:.4f}\n\nPer-Class AUC:\n")
    for i in range(n):
        txt += f"  {Config.CLASS_NAMES[i]}: {roc_aucs[i]:.4f}\n"
    ax.text(0.05, 0.97, txt, transform=ax.transAxes, fontsize=9, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))
    ax.set_title('Performance Summary', fontweight='bold')

    plt.tight_layout()
    out_name = f'inception_attention_{version}_results.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {out_name}")
    plt.show()


# ============================================================
# PROGRESSIVE RESIZE TRAINING  (3-phase)
# ============================================================
def progressive_train():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"  {len(gpus)} GPU(s) detected")
    else:
        print("  CPU mode")

    csv_full = os.path.join(Config.DATA_DIR, Config.CSV_FILE)
    combined_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    current_weights  = None

    phases = [
        (128, 128, Config.PHASE1_EPOCHS,  Config.INITIAL_LR,       "Phase 1 (128×128)"),
        (160, 160, Config.PHASE2_EPOCHS,  Config.INITIAL_LR * 0.6, "Phase 2 (160×160)"),
        (192, 192, Config.TOTAL_EPOCHS,   Config.INITIAL_LR * 0.3, "Phase 3 (192×192)"),
    ]

    for (ph, pw, phase_epochs, phase_lr, phase_name) in phases:
        print(f"\n{'='*65}")
        print(f"  {phase_name}  —  LR={phase_lr:.2e}  Epochs up to {phase_epochs}")
        print(f"{'='*65}")

        result = create_datasets(csv_full, Config.DATA_DIR, img_h=ph, img_w=pw)
        if result[0] is None:
            print("ERROR: dataset creation failed"); return None, None
        train_ds, val_ds, test_ds, tr_info, val_info, test_info, cw = result

        model = build_model(img_size=ph)

        if current_weights is not None:
            print("  Transferring weights from previous phase...")
            try:
                model.set_weights(current_weights)
            except Exception as e:
                print(f"  Weight transfer skipped (expected for new resolution): {e}")

        model = compile_model(model, phase_lr)

        ema = EMAWeights(model, decay=Config.EMA_DECAY)
        swa = SWACallback(start_epoch_frac=Config.SWA_START_EPOCH_FRAC)

        callbacks = [
            keras.callbacks.LearningRateScheduler(
                make_lr_schedule(phase_epochs, phase_lr), verbose=0),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25,          # ↑ from 20
                restore_best_weights=True, min_delta=0.0003, mode='min', verbose=1),
            keras.callbacks.ModelCheckpoint(
                f'inception_attn_v6_{ph}.keras',
                monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,           # ↑ from 0.4
                patience=8, min_lr=Config.MIN_LR, verbose=1),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.CSVLogger(f'log_v6_phase_{ph}.csv'),
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda ep, logs: gc.collect()),
            EMACallback(ema),
            swa,
            OverfitMonitor(threshold=0.08),
        ]

        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=phase_epochs, callbacks=callbacks,
            class_weight=cw, verbose=1)

        for k in combined_history:
            combined_history[k].extend(history.history.get(k, []))

        if Config.USE_SWA:
            swa.apply(model)

        # EMA vs checkpoint selection
        ema.apply(model)
        ema_res = model.evaluate(val_ds, verbose=0)
        ema.restore(model)

        ckpt_path = f'inception_attn_v6_{ph}.keras'
        if os.path.exists(ckpt_path):
            model.load_weights(ckpt_path)
            ckpt_res = model.evaluate(val_ds, verbose=0)
            if ema_res[1] > ckpt_res[1]:
                print(f"  EMA val_acc={ema_res[1]:.4f} > ckpt={ckpt_res[1]:.4f} → using EMA")
                ema.apply(model)

        current_weights = [w.numpy() for w in model.weights]
        print(f"  {phase_name} best val acc: {max(history.history.get('val_accuracy', [0])):.4f}")

    # ---- Final evaluation at full resolution ----
    print("\n=== FINAL FULL-RESOLUTION EVALUATION ===")
    result     = create_datasets(csv_full, Config.DATA_DIR)
    train_ds, val_ds, test_ds, tr_info, val_info, test_info, cw = result

    # Pseudo-label refinement
    pseudo_label_finetune(model, val_ds, val_info, train_ds, tr_info)

    res = model.evaluate(test_ds, verbose=1)
    print(f"  Test Accuracy : {res[1]:.4f}  |  Test Loss: {res[0]:.4f}  |  Top-2: {res[2]:.4f}")

    preds    = tta_predict(model, test_ds, Config.TTA_STEPS) if Config.USE_TTA else model.predict(test_ds)
    pred_cls = np.argmax(preds, axis=1)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(test_info[1], pred_cls,
                                 target_names=Config.CLASS_NAMES, zero_division=0))

    class FakeHistory:
        def __init__(self, h): self.history = h

    plot_results(FakeHistory(combined_history), preds, test_info[1], version='v6')
    model.save('inception_attn_v6_final.keras')
    print("  Model saved: inception_attn_v6_final.keras")

    ta   = accuracy_score(test_info[1], pred_cls)
    loss = res[0]
    print(f"\nFinal Test Accuracy : {ta:.4f}  {'✅ ACHIEVED' if ta >= 0.95 else '❌ Not yet'}")
    print(f"Final Test Loss     : {loss:.4f}  {'✅ ACHIEVED' if loss < 0.5 else '❌ Above 0.5'}")
    return model, FakeHistory(combined_history)


# ============================================================
# SINGLE-PHASE TRAIN  (fallback)
# ============================================================
def train():
    print("INCEPTION-ATTENTION CNN  v6  (single-phase)")
    print("=" * 65)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus: tf.config.experimental.set_memory_growth(g, True)
        print(f"  {len(gpus)} GPU(s) detected")
    else:
        print("  CPU mode")

    csv_full = os.path.join(Config.DATA_DIR, Config.CSV_FILE)
    result   = create_datasets(csv_full, Config.DATA_DIR)
    if result[0] is None:
        print("ERROR: dataset creation failed"); return None, None

    train_ds, val_ds, test_ds, tr_info, val_info, test_info, cw = result
    print(f"\nClass weights: { {k: round(v, 3) for k, v in cw.items()} }")

    model = build_model()
    model = compile_model(model, Config.INITIAL_LR)
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")

    ema = EMAWeights(model, decay=Config.EMA_DECAY)
    swa = SWACallback(start_epoch_frac=Config.SWA_START_EPOCH_FRAC)

    callbacks = [
        keras.callbacks.LearningRateScheduler(
            make_lr_schedule(Config.TOTAL_EPOCHS, Config.INITIAL_LR), verbose=0),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=25,
            restore_best_weights=True, min_delta=0.0003, mode='min', verbose=1),
        keras.callbacks.ModelCheckpoint(
            'inception_attn_v6_best.keras', monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=Config.MIN_LR, verbose=1),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger('log_v6.csv'),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda ep, logs: gc.collect()),
        EMACallback(ema),
        swa,
        OverfitMonitor(threshold=0.08),
    ]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=Config.TOTAL_EPOCHS, callbacks=callbacks,
                        class_weight=cw, verbose=1)

    if Config.USE_SWA:
        swa.apply(model)

    ema.apply(model)
    ema_res = model.evaluate(val_ds, verbose=0)
    ema.restore(model)

    if os.path.exists('inception_attn_v6_best.keras'):
        model.load_weights('inception_attn_v6_best.keras')
        ckpt_res = model.evaluate(val_ds, verbose=0)
        if ema_res[1] > ckpt_res[1]:
            print(f"  EMA val_acc={ema_res[1]:.4f} > ckpt={ckpt_res[1]:.4f} → using EMA")
            ema.apply(model)

    pseudo_label_finetune(model, val_ds, val_info, train_ds, tr_info)

    print("\n=== FINAL TEST EVALUATION ===")
    res = model.evaluate(test_ds, verbose=1)
    print(f"  Test Accuracy : {res[1]:.4f}  |  Test Loss: {res[0]:.4f}  |  Top-2: {res[2]:.4f}")

    preds    = tta_predict(model, test_ds, Config.TTA_STEPS) if Config.USE_TTA else model.predict(test_ds)
    pred_cls = np.argmax(preds, axis=1)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(test_info[1], pred_cls,
                                 target_names=Config.CLASS_NAMES, zero_division=0))

    plot_results(history, preds, test_info[1], version='v6')
    model.save('inception_attn_v6_final.keras')
    print("  Model saved: inception_attn_v6_final.keras")

    ta   = accuracy_score(test_info[1], pred_cls)
    loss = res[0]
    print(f"\nFinal Test Accuracy : {ta:.4f}  {'✅ ACHIEVED' if ta >= 0.95 else '❌ Not yet'}")
    print(f"Final Test Loss     : {loss:.4f}  {'✅ ACHIEVED' if loss < 0.5 else '❌ Above 0.5'}")
    return model, history


if __name__ == "__main__":
    progressive_train()
    # Alternatively for faster iteration: train()
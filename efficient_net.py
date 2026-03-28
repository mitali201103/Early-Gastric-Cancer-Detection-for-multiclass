import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import layers
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import random, gc

print(f"Keras: {keras.__version__} | TF: {tf.__version__}")
print(f"GPUs: {len(tf.config.list_physical_devices('GPU'))}")

DATA_DIR    = "/kaggle/input/datasets/mitali201103/gastric-cancer-data/processed_multiclass/processed_images_multiclass"
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_CLASSES = 4

# Balance
random.seed(42)
records = []
for cls in range(4):
    cls_dir = os.path.join(DATA_DIR, f'class_{cls}')
    files   = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    sel     = random.sample(files, 4000) if len(files)>=4000 else random.choices(files, k=4000)
    for f in sel:
        records.append({'filepath': os.path.join(cls_dir, f), 'label': cls})
    print(f"class_{cls}: {len(files)} → 4000")

df       = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
print(f"Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")

def make_dataset(df, training=True):
    paths  = df["filepath"].values
    labels = keras.utils.to_categorical(df["label"].values, NUM_CLASSES)
    ds     = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0, 1)
        return img, label
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training: ds = ds.shuffle(1000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, True)
val_ds   = make_dataset(val_df,   False)
test_ds  = make_dataset(test_df,  False)

strategy = tf.distribute.MirroredStrategy()
print(f"✅ Using {strategy.num_replicas_in_sync} GPU(s)")

with strategy.scope():
    base = keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.3)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model   = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])

print("✅ Model ready! Starting Phase 1...")

cbs = [
    keras.callbacks.ModelCheckpoint('/kaggle/working/best.keras',
        monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
        patience=3, min_lr=1e-7, verbose=1),
]

# Phase 1 — frozen
h1 = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cbs, verbose=1)

# Phase 2 — unfreeze
print("\n=== Phase 2: Fine-tuning ===")
with strategy.scope():
    base.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])

h2 = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=cbs, verbose=1)

# Combine history
all_acc     = h1.history['accuracy']     + h2.history['accuracy']
all_val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
all_loss    = h1.history['loss']         + h2.history['loss']
all_val_loss= h1.history['val_loss']     + h2.history['val_loss']

# Test
print("\n=== Test Evaluation ===")
model.load_weights('/kaggle/working/best.keras')
res = model.evaluate(test_ds, verbose=1)
print(f"🎯 Test Accuracy: {res[1]:.4f}")
print(f"📉 Test Loss:     {res[0]:.4f}")

# Plots
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_score,
                             recall_score, f1_score, accuracy_score)
from sklearn.preprocessing import label_binarize

CLASS_NAMES   = ['Normal','Early','Intermediate','Advanced']
y_pred_probs  = model.predict(test_ds, verbose=0)
y_pred        = np.argmax(y_pred_probs, axis=1)
y_true        = test_df["label"].values[:len(y_pred)]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Gastric Cancer Detection Results', fontsize=16, fontweight='bold')

# Confusion matrix
ax = axes[0,0]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ax.set_title('Confusion Matrix', fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# ROC
ax = axes[0,1]
y_bin   = label_binarize(y_true, classes=[0,1,2,3])
colors  = ['blue','red','green','orange']
for i,(cls,c) in enumerate(zip(CLASS_NAMES,colors)):
    fpr,tpr,_ = roc_curve(y_bin[:,i], y_pred_probs[:,i])
    ax.plot(fpr, tpr, color=c, lw=2, label=f'{cls} AUC={auc(fpr,tpr):.3f}')
ax.plot([0,1],[0,1],'k--'); ax.set_title('ROC Curves',fontweight='bold')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Precision/Recall/F1
ax  = axes[0,2]
pre = precision_score(y_true,y_pred,average=None,zero_division=0)
rec = recall_score(y_true,y_pred,average=None,zero_division=0)
f1s = f1_score(y_true,y_pred,average=None,zero_division=0)
x   = np.arange(4); w = 0.25
ax.bar(x-w, pre, w, label='Precision', color='steelblue', alpha=0.8)
ax.bar(x,   rec, w, label='Recall',    color='coral',     alpha=0.8)
ax.bar(x+w, f1s, w, label='F1',        color='green',     alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES,rotation=15)
ax.set_ylim([0,1.15]); ax.legend(); ax.grid(alpha=0.3,axis='y')
ax.set_title('Per-Class Metrics', fontweight='bold')

# Accuracy curve
ax = axes[1,0]
ax.plot(all_acc,     'b-', lw=2, label='Train')
ax.plot(all_val_acc, 'r-', lw=2, label='Val')
ax.axhline(0.95, color='green', ls='--', label='95% target')
ax.set_title('Accuracy Curve', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel('Epoch')

# Loss curve
ax = axes[1,1]
ax.plot(all_loss,     'b-', lw=2, label='Train')
ax.plot(all_val_loss, 'r-', lw=2, label='Val')
ax.axhline(0.5, color='red', ls='--', label='0.5 target')
ax.set_title('Loss Curve', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel('Epoch')

# Summary
ax = axes[1,2]; ax.axis('off')
acc_s = accuracy_score(y_true, y_pred)
summary = (f"EfficientNetB3 Fine-tuned\n\n"
           f"Test Accuracy : {acc_s:.4f} {'✅' if acc_s>=0.95 else '🔄'}\n"
           f"Macro F1      : {f1_score(y_true,y_pred,average='macro',zero_division=0):.4f}\n\n"
           f"Per-Class Accuracy:\n")
for i,cls in enumerate(CLASS_NAMES):
    m = y_true==i
    summary += f"  {cls}: {accuracy_score(y_true[m],y_pred[m]):.4f}\n"
ax.text(0.05,0.95,summary,transform=ax.transAxes,fontsize=11,va='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.8))
ax.set_title('Summary', fontweight='bold')

plt.tight_layout()
plt.savefig('/kaggle/working/results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plot saved!")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

model.save('/kaggle/working/gastric_final.keras')
print("✅ Model saved!")
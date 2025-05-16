import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, regularizers, Model
import seaborn as sns
from PIL import Image

# Define custom loss function
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=3.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        y_pred_true = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
        focal_weight = tf.pow(1.0 - y_pred_true, self.gamma)
        focal_loss = focal_weight * ce_loss
        if self.alpha is not None:
            alpha_weight = tf.gather(self.alpha, y_true)
            focal_loss = alpha_weight * focal_loss
        return tf.reduce_mean(focal_loss)

# Data directory
data_dir = 'Image_classification_data'
img_dir = os.path.join(data_dir, 'patch_images')

# Load dataset
main_data = pd.read_csv(os.path.join(data_dir, 'data_labels_mainData.csv'))

# Patient-wise initial split (70% train, 15% val, 15% test)
patient_ids = main_data['patientID'].unique()
train_ids, temp_ids = train_test_split(patient_ids, train_size=0.7, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Define datasets
train_data = main_data[main_data['patientID'].isin(train_ids)].reset_index(drop=True)
val_data = main_data[main_data['patientID'].isin(val_ids)].reset_index(drop=True)
test_data = main_data[main_data['patientID'].isin(test_ids)].reset_index(drop=True)
train_val_data = main_data[main_data['patientID'].isin(np.concatenate([train_ids, val_ids]))].reset_index(drop=True)

# Define dataset creation function
def make_ds(df, label_col, n_cls, training=True):
    def parse_image(filename, label):
        image_path = tf.strings.join([img_dir + '/', filename])
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [27, 27]) / 255.0
        return image, label

    filenames = df['ImageName'].values
    labels = df[label_col].values
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(1000).batch(64).repeat()
    else:
        dataset = dataset.batch(64)
    return dataset.prefetch(tf.data.AUTOTUNE)

# Define data augmentation
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1)
])

# Define SimpleCNN_Shallow for both tasks
def simple_cnn_shallow(n_classes, name="SimpleCNN_Shallow"):
    model = tf.keras.Sequential([
        layers.Input(shape=(27, 27, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='sigmoid' if n_classes == 1 else 'softmax')
    ], name=name)
    return model

# Define SimpleCNN_Deep_Binary (kept for reference but not used in final evaluation)
def simple_cnn_deep_binary(name="SimpleCNN_Deep_Binary"):
    x = inp = layers.Input((27, 27, 3))
    x = layers.Rescaling(1./255)(x)
    x = data_aug(x, training=True)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-3))(x)
    return Model(inp, out, name=name)

# Define evaluation functions
def evaluate_two_stage(model_multi, model_bin, test_ds, threshold=0.5, name="Two-Stage Model"):
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        cell_type_preds = model_multi.predict(images, verbose=0)
        cell_types = np.argmax(cell_type_preds, axis=1)
        bin_preds = model_bin.predict(images, verbose=0)
        for i in range(len(cell_types)):
            if cell_types[i] != 2:
                y_pred.append(0)
            else:
                y_pred.append(1 if bin_preds[i][0] >= threshold else 0)
            y_true.append(labels.numpy()[i])
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"{name} - Test Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return accuracy, macro_f1, cm

def evaluate_multiclass_model(model, ds, name):
    y_true = []
    y_pred = []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"{name} - Test Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    return accuracy, macro_f1, cm

# Cross-validation setup
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
patient_labels_df = train_val_data.groupby('patientID').agg({'isCancerous': 'mean'}).reset_index()
train_val_patient_ids = patient_labels_df['patientID'].values
patient_labels = (patient_labels_df['isCancerous'] > 0).astype(int).values

# Binary Classification
print("### Binary Classification (isCancerous) with Cross-Validation")
binary_scores = {'accuracy': [], 'f1': []}
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_patient_ids, patient_labels)):
    fold_train_ids = train_val_patient_ids[train_idx]
    fold_val_ids = train_val_patient_ids[val_idx]
    fold_train_data = train_val_data[train_val_data['patientID'].isin(fold_train_ids)].reset_index(drop=True)
    fold_val_data = train_val_data[train_val_data['patientID'].isin(fold_val_ids)].reset_index(drop=True)

    bin_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=fold_train_data['isCancerous'])
    class_weights = {0: bin_weights[0], 1: bin_weights[1]}

    train_ds = make_ds(fold_train_data, 'isCancerous', 2, True)
    val_ds = make_ds(fold_val_data, 'isCancerous', 2, False)

    model_bin = simple_cnn_shallow(1, f"SimpleCNN_Shallow_Bin_Fold{fold}")
    model_bin.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=BinaryFocalCrossentropy(alpha=0.75, gamma=2.0),
        metrics=['accuracy', tf.keras.metrics.Recall()]
    )

    history = model_bin.fit(
        train_ds, steps_per_epoch=len(fold_train_data) // 64,
        validation_data=val_ds, validation_steps=len(fold_val_data) // 64,
        epochs=50, verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    )

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model_bin.predict(images, verbose=0)
        y_pred.extend((preds > 0.5).astype(int).flatten())
        y_true.extend(labels.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    binary_scores['accuracy'].append(acc)
    binary_scores['f1'].append(f1)
    print(f"Fold {fold + 1}: Accuracy = {acc:.4f}, Macro-F1 = {f1:.4f}")

print(f"Binary CV - Mean Accuracy: {np.mean(binary_scores['accuracy']):.4f}, Mean Macro-F1: {np.mean(binary_scores['f1']):.4f}")

# Multiclass Classification
print("\n### Multiclass Classification (cellType) with Cross-Validation")
multi_scores = {'accuracy': [], 'f1': []}
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_patient_ids, patient_labels)):
    fold_train_ids = train_val_patient_ids[train_idx]
    fold_val_ids = train_val_patient_ids[val_idx]
    fold_train_data = train_val_data[train_val_data['patientID'].isin(fold_train_ids)].reset_index(drop=True)
    fold_val_data = train_val_data[train_val_data['patientID'].isin(fold_val_ids)].reset_index(drop=True)

    multi_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=fold_train_data['cellType'])
    class_weights = dict(enumerate(multi_weights))

    train_ds = make_ds(fold_train_data, 'cellType', 4, True)
    val_ds = make_ds(fold_val_data, 'cellType', 4, False)

    model_multi = simple_cnn_shallow(4, f"SimpleCNN_Shallow_Multi_Fold{fold}")
    model_multi.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model_multi.fit(
        train_ds, steps_per_epoch=len(fold_train_data) // 64,
        validation_data=val_ds, validation_steps=len(fold_val_data) // 64,
        epochs=50, verbose=0,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        class_weight=class_weights
    )

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model_multi.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    multi_scores['accuracy'].append(acc)
    multi_scores['f1'].append(f1)
    print(f"Fold {fold + 1}: Accuracy = {acc:.4f}, Macro-F1 = {f1:.4f}")

print(f"Multiclass CV - Mean Accuracy: {np.mean(multi_scores['accuracy']):.4f}, Mean Macro-F1: {np.mean(multi_scores['f1']):.4f}")

# Final Test Evaluation
print("\n### Final Test Evaluation")

# Compute alpha for binary focal loss based on class distribution
pos_count = np.sum(train_data['isCancerous'] == 1)
total_count = len(train_data)
alpha_bin = pos_count / total_count

# Compute class weights for multiclass
multi_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), y=train_data['cellType'])
multi_class_weights = dict(enumerate(multi_weights))

# Create datasets
train_ds_bin = make_ds(train_data, 'isCancerous', 2, True)
val_ds_bin = make_ds(val_data, 'isCancerous', 2, False)
test_ds_bin = make_ds(test_data, 'isCancerous', 2, False)

train_ds_multi = make_ds(train_data, 'cellType', 4, True)
val_ds_multi = make_ds(val_data, 'cellType', 4, False)
test_ds_multi = make_ds(test_data, 'cellType', 4, False)

# Train final binary model
final_model_bin = simple_cnn_shallow(1, "Final_SimpleCNN_Shallow_Bin")
final_model_bin.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=BinaryFocalCrossentropy(alpha=alpha_bin, gamma=2.0),
    metrics=['accuracy', tf.keras.metrics.Recall()]
)
history_bin = final_model_bin.fit(
    train_ds_bin, steps_per_epoch=len(train_data) // 64,
    validation_data=val_ds_bin, validation_steps=len(val_data) // 64,
    epochs=50, verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)
final_model_bin.save('final_model_bin_shallow.h5')
print("Saved final binary model as 'final_model_bin_shallow.h5'")

# Train and save final multiclass model
print("Training new multiclass model")
final_model_multi = simple_cnn_shallow(4, "Final_SimpleCNN_Shallow_Multi")
final_model_multi.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
history_multi = final_model_multi.fit(
    train_ds_multi, steps_per_epoch=len(train_data) // 64,
    validation_data=val_ds_multi, validation_steps=len(val_data) // 64,
    epochs=50, verbose=1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    class_weight=multi_class_weights
)
final_model_multi.save('final_model_multi_shallow.h5')
print("Saved final multiclass model as 'final_model_multi_shallow.h5'")

# Load the newly trained multiclass model
final_model_multi = tf.keras.models.load_model('final_model_multi_shallow.h5')
print("Loaded newly trained multiclass model from 'final_model_multi_shallow.h5'")

# Evaluate models
acc_bin, f1_bin, cm_bin = evaluate_two_stage(final_model_multi, final_model_bin, test_ds_bin, 0.5, "Final Binary Model")
acc_multi, f1_multi, cm_multi = evaluate_multiclass_model(final_model_multi, test_ds_multi, "Final Multiclass Model")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Epithelial', 'Inflammatory', 'Fibroblast', 'Others'],
            yticklabels=['Epithelial', 'Inflammatory', 'Fibroblast', 'Others'])
plt.title('Confusion Matrix - Final Multiclass Model (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('final_multiclass_confusion_matrix.png')
"""Rescue tool, not needed for the normal pipeline: ../kaggle_train.py now writes checkpoints
directly in Keras 2's legacy format (save_legacy_keras2_weights), so freshly trained checkpoints
never need this. Kept for converting any checkpoint saved with Keras 3's own model.save_weights()
(e.g. from before that fix, kernel versions <=8) after the fact.

Converts a Keras-3-saved CRNNxCTC weights-only .h5 file (native format: layers/<generic_name>/
vars/<idx>) into Keras 2's legacy HDF5 weights format (root 'layer_names' attr + per-layer named
group with 'weight_names' attr + datasets), which this project's local TF 2.10 pipeline's
Model.load_weights() expects.

Usage: python convert_keras3_checkpoint.py <keras3_input.h5> <keras2_output.h5>

Mapping derived from nomnaocr_lib/layers.py + model.py's exact, deterministic layer-construction
order (custom_cnn's per-block per-conv loop: Conv2D, BatchNormalization, Activation[, MaxPooling2D]
repeated, then reshape_features, bigru1, bigru2, dense) cross-checked against the actual shapes
found in the downloaded file (e.g. batch_normalization sizes 64,128,256,256,512,512,512,512 and
conv2d kernel shapes (3,3,3,64)...(2,2,512,512) match this project's block1..block5 config exactly
- not guessed blind). Keras's default auto-naming assigns the bare class name to the first
instance and "_N" to the Nth-following instance, in GLOBAL model-construction order, which is
exactly nomnaocr_lib's own layer order.

Only WEIGHTED layers need an entry - Activation/MaxPooling2D/Reshape/InputLayer have none, and
Model.load_weights()'s positional matching (no by_name=True) only counts layers with weights,
which is exactly the 19 entries below (matching the "Model expected 19 layers" count from the
original error before this converter existed).
"""
import sys

import h5py

# (our_layer_name, keras3_generic_group) for the 19 weighted layers, in build_crnn's construction order
CONV_BN_ORDER = [
    ("block1_conv1", "block1_bn1", "conv2d", "batch_normalization"),
    ("block2_conv1", "block2_bn1", "conv2d_1", "batch_normalization_1"),
    ("block3_conv1", "block3_bn1", "conv2d_2", "batch_normalization_2"),
    ("block3_conv2", "block3_bn2", "conv2d_3", "batch_normalization_3"),
    ("block4_conv1", "block4_bn1", "conv2d_4", "batch_normalization_4"),
    ("block4_conv2", "block4_bn2", "conv2d_5", "batch_normalization_5"),
    ("block5_conv1", "block5_bn1", "conv2d_6", "batch_normalization_6"),
    ("block5_conv2", "block5_bn2", "conv2d_7", "batch_normalization_7"),
]


def read_layer_vars(src, group_path):
    g = src[group_path]["vars"] if "vars" in src[group_path] else src[group_path]
    idxs = sorted((int(k) for k in g.keys()), key=int)
    return [g[str(i)][()] for i in idxs]


def convert(src_path, dst_path):
    with h5py.File(src_path, "r") as src:
        layers_grp = src["layers"]
        ordered = []  # list of (our_name, [weight_arrays])

        for conv_name, bn_name, conv_k3, bn_k3 in CONV_BN_ORDER:
            ordered.append((conv_name, read_layer_vars(layers_grp, conv_k3)))
            ordered.append((bn_name, read_layer_vars(layers_grp, bn_k3)))

        for our_name, k3_name in [("bigru1", "bidirectional"), ("bigru2", "bidirectional_1")]:
            fwd = read_layer_vars(layers_grp, f"{k3_name}/forward_layer/cell")
            bwd = read_layer_vars(layers_grp, f"{k3_name}/backward_layer/cell")
            ordered.append((our_name, fwd + bwd))

        ordered.append(("rnn_output", read_layer_vars(layers_grp, "dense")))

        # Sanity: expected shapes per layer type, verified against nomnaocr_lib/model.py's
        # build_crnn() architecture before trusting this conversion.
        assert len(ordered) == 19, f"expected 19 weighted layers, got {len(ordered)}"
        conv0_kernel_shape = ordered[0][1][0].shape
        assert conv0_kernel_shape == (3, 3, 3, 64), f"block1_conv1 kernel shape mismatch: {conv0_kernel_shape}"
        dense_kernel_shape = ordered[-1][1][0].shape
        assert dense_kernel_shape[0] == 512, f"rnn_output kernel shape mismatch: {dense_kernel_shape}"
        bigru1_shapes = [w.shape for w in ordered[16][1]]
        assert len(bigru1_shapes) == 6, f"bigru1 should have 6 weight arrays (fwd+bwd x3), got {len(bigru1_shapes)}"

    with h5py.File(dst_path, "w") as dst:
        dst.attrs["layer_names"] = [name.encode("utf8") for name, _ in ordered]
        for name, weights in ordered:
            g = dst.create_group(name)
            weight_names = [f"{name}_w{i}".encode("utf8") for i in range(len(weights))]
            g.attrs["weight_names"] = weight_names
            for wname, arr in zip(weight_names, weights):
                g.create_dataset(wname.decode("utf8"), data=arr)

    print(f"Converted {len(ordered)} layers -> {dst_path}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])

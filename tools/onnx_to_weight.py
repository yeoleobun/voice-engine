import onnx
import numpy as np
import struct
import sys
from onnx import numpy_helper

def extract_weights(onnx_path, output_path):
    print(f"Loading model from {onnx_path}")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # Find the If node for 16k branch
    if_node = None
    for node in graph.node:
        if node.op_type == 'If':
            if_node = node
            break
    
    target_graph = graph
    if if_node:
        print("Found If node, extracting 16kHz branch (then_branch)")
        for attr in if_node.attribute:
            if attr.name == 'then_branch':
                target_graph = attr.g
                break
    
    weights = {}
    # Extract initializers from the target graph
    for tensor in target_graph.initializer:
        weights[tensor.name] = numpy_helper.to_array(tensor)
        
    # Also check main graph initializers as they might be shared
    for tensor in graph.initializer:
        if tensor.name not in weights:
             weights[tensor.name] = numpy_helper.to_array(tensor)

    # Also check Constant nodes in the target graph
    for node in target_graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    tensor = attr.t
                    weights[node.output[0]] = numpy_helper.to_array(tensor)

    print(f"Found {len(weights)} tensors.")
    for name, arr in weights.items():
        print(f"  {name}: {arr.shape}")
    
    # Map to our naming convention
    # We need to find the specific tensors based on their shapes or names
    
    mapped_weights = {}
    
    # Helper to find by shape
    def find_by_shape(shape):
        candidates = []
        for name, arr in weights.items():
            if list(arr.shape) == list(shape):
                candidates.append(name)
        return candidates

    # 1. STFT
    # Shape: [258, 1, 256]
    stft_candidates = find_by_shape([258, 1, 256])
    if stft_candidates:
        print(f"Found STFT weights: {stft_candidates[0]}")
        mapped_weights["stft_weight"] = weights[stft_candidates[0]]

    # 2. Encoder
    # Layer 0: [128, 129, 3]
    enc0_w = find_by_shape([128, 129, 3])
    if enc0_w:
        print(f"Found Encoder 0 weights: {enc0_w[0]}")
        mapped_weights["enc0_weight"] = weights[enc0_w[0]]
        # Look for bias [128]
        enc0_b = find_by_shape([128])
        # There might be multiple [128] biases. Need to be careful.
        # Usually the bias name is related to weight name.
        # e.g. ...weight -> ...bias
        base_name = enc0_w[0].replace("weight", "bias")
        if base_name in weights:
             mapped_weights["enc0_bias"] = weights[base_name]
        else:
             # Try to find by node connection if needed, but let's try simple name matching first
             pass

    # Layer 1: [64, 128, 3]
    enc1_w = find_by_shape([64, 128, 3])
    if enc1_w:
        print(f"Found Encoder 1 weights: {enc1_w[0]}")
        mapped_weights["enc1_weight"] = weights[enc1_w[0]]
        base_name = enc1_w[0].replace("weight", "bias")
        if base_name in weights: mapped_weights["enc1_bias"] = weights[base_name]

    # Layer 2: [64, 64, 3]
    enc2_w = find_by_shape([64, 64, 3])
    if enc2_w:
        print(f"Found Encoder 2 weights: {enc2_w[0]}")
        mapped_weights["enc2_weight"] = weights[enc2_w[0]]
        base_name = enc2_w[0].replace("weight", "bias")
        if base_name in weights: mapped_weights["enc2_bias"] = weights[base_name]

    # Layer 3: [128, 64, 3]
    enc3_w = find_by_shape([128, 64, 3])
    if enc3_w:
        print(f"Found Encoder 3 weights: {enc3_w[0]}")
        mapped_weights["enc3_weight"] = weights[enc3_w[0]]
        base_name = enc3_w[0].replace("weight", "bias")
        if base_name in weights: mapped_weights["enc3_bias"] = weights[base_name]

    # 3. LSTM
    # Weights: [512, 128] (4*128, 128)
    # There should be two layers, so 4 weight matrices in total.
    lstm_w = find_by_shape([512, 128])
    if len(lstm_w) >= 4:
        print(f"Found LSTM weights (2 layers): {lstm_w}")
        lstm_sorted = sorted(lstm_w)
        # Assuming sorted order:
        # decoder.rnn.weight_hh
        # decoder.rnn.weight_ih
        # decoder.rnn_1.weight_hh
        # decoder.rnn_1.weight_ih
        
        # Layer 0
        # Note: 'hh' comes before 'ih' alphabetically? No, h < i.
        # But let's check if they have 'rnn' vs 'rnn_1'.
        
        layer0 = [w for w in lstm_sorted if 'rnn_1' not in w]
        layer1 = [w for w in lstm_sorted if 'rnn_1' in w]
        
        if len(layer0) == 2 and len(layer1) == 2:
            # Sort by name to separate hh and ih
            # usually ...weight_hh and ...weight_ih
            # hh comes before ih
            layer0.sort()
            layer1.sort()
            
            mapped_weights["lstm_w_hh"] = weights[layer0[0]]
            mapped_weights["lstm_w_ih"] = weights[layer0[1]]
            
            mapped_weights["lstm_w_hh_l1"] = weights[layer1[0]]
            mapped_weights["lstm_w_ih_l1"] = weights[layer1[1]]
        else:
            print("Warning: Could not separate LSTM layers by name 'rnn_1'")
            # Fallback to simple sort
            mapped_weights["lstm_w_hh"] = weights[lstm_sorted[0]]
            mapped_weights["lstm_w_ih"] = weights[lstm_sorted[1]]
            mapped_weights["lstm_w_hh_l1"] = weights[lstm_sorted[2]]
            mapped_weights["lstm_w_ih_l1"] = weights[lstm_sorted[3]]

    elif len(lstm_w) >= 2:
        # ... existing single layer logic ...
        print(f"Found LSTM weights (1 layer?): {lstm_w}")
        lstm_sorted = sorted(lstm_w)
        mapped_weights["lstm_w_hh"] = weights[lstm_sorted[0]]
        mapped_weights["lstm_w_ih"] = weights[lstm_sorted[1]]
            
    # LSTM Bias: [512]
    # There should be two layers: bias_ih, bias_hh for each
    lstm_b = find_by_shape([512])
    if len(lstm_b) >= 4:
        print(f"Found LSTM biases (2 layers): {lstm_b}")
        lstm_b_sorted = sorted(lstm_b)
        
        layer0 = [b for b in lstm_b_sorted if 'rnn_1' not in b]
        layer1 = [b for b in lstm_b_sorted if 'rnn_1' in b]
        
        if len(layer0) == 2 and len(layer1) == 2:
            layer0.sort()
            layer1.sort()
            mapped_weights["lstm_b_hh"] = weights[layer0[0]]
            mapped_weights["lstm_b_ih"] = weights[layer0[1]]
            mapped_weights["lstm_b_hh_l1"] = weights[layer1[0]]
            mapped_weights["lstm_b_ih_l1"] = weights[layer1[1]]
        else:
            mapped_weights["lstm_b_hh"] = weights[lstm_b_sorted[0]]
            mapped_weights["lstm_b_ih"] = weights[lstm_b_sorted[1]]
            mapped_weights["lstm_b_hh_l1"] = weights[lstm_b_sorted[2]]
            mapped_weights["lstm_b_ih_l1"] = weights[lstm_b_sorted[3]]

    elif len(lstm_b) >= 2:
        print(f"Found LSTM biases (1 layer?): {lstm_b}")
        lstm_b_sorted = sorted(lstm_b)
        mapped_weights["lstm_b_hh"] = weights[lstm_b_sorted[0]]
        mapped_weights["lstm_b_ih"] = weights[lstm_b_sorted[1]]


    # 4. Output Dense (Conv1d 1x1)
    # Shape: [1, 128, 1]
    out_w = find_by_shape([1, 128, 1])
    if out_w:
        print(f"Found Output weights: {out_w[0]}")
        mapped_weights["out_weight"] = weights[out_w[0]]
        base_name = out_w[0].replace("weight", "bias")
        if base_name in weights: mapped_weights["out_bias"] = weights[base_name]

    print(f"Mapped {len(mapped_weights)} tensors.")
    
    # Write to binary file
    with open(output_path, 'wb') as f:
        # Write number of tensors
        f.write(struct.pack('<I', len(mapped_weights)))
        
        for name, arr in mapped_weights.items():
            # Write name length and name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write shape length and shape
            f.write(struct.pack('<I', len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack('<I', dim))
                
            # Write data length and data (float32)
            flat_data = arr.astype(np.float32).flatten()
            f.write(struct.pack('<I', len(flat_data) * 4))
            f.write(flat_data.tobytes())
            
    print(f"Saved weights to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 onnx_to_bin.py <onnx_file> <output_bin>")
        sys.exit(1)
        
    extract_weights(sys.argv[1], sys.argv[2])

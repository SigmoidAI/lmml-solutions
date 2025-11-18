import torch
import numpy as np
from transformers import T5ForConditionalGeneration

def extract_flag_from_model(model: torch.nn.Module) -> str:
    """Extract the hidden flag from the LSBs of a chosen layer's weights.

    Here we use the first parameter returned by model.parameters().
    You can change which layer you use if needed.
    """
    # 1. Take a parameter tensor (here: first parameter)
    layer = next(model.parameters())

    # 2. Flatten and move to CPU as float32
    weights = layer.data.view(-1).cpu().numpy().astype(np.float32)

    # 3. Reinterpret float32 values as uint32 bit patterns
    weights_int = weights.view(np.uint32)

    # 4. Extract least significant bits (LSBs)
    lsb_bits = weights_int & 1  # array of 0/1

    # 5. Keep only a length divisible by 8 (full bytes)
    usable_len = len(lsb_bits) - (len(lsb_bits) % 8)
    usable_bits = lsb_bits[:usable_len]

    # 6. Reshape bits into [num_bytes, 8]
    bit_matrix = usable_bits.reshape(-1, 8)

    # 7. Convert each group of 8 bits to one byte
    byte_values = np.packbits(bit_matrix, axis=1, bitorder="big").flatten()

    # 8. Try to decode all bytes as UTF-8
    try:
        return byte_values.tobytes().decode("utf-8")
    except UnicodeDecodeError:
        return "<Failed to decode flag>"

if __name__ == "__main__":
    # Load the model with the hidden flag
    loaded_model = T5ForConditionalGeneration.from_pretrained(
        "CezarCalin/hidden-flag-model"
    )

    # Extract and print the flag
    extracted = extract_flag_from_model(loaded_model)
    print("Extracted flag:", extracted)

# Neural Cypher

- [Task - Description](#task)
- [Solution - Explanation](#solution)

## Task

### Task Overview

#### 📖 Storyline
Storyline: "The Model Was Never Meant to Help"

A brilliant but reclusive machine learning researcher open-sourced two powerful models, claiming they would revolutionize their respective domains. First, a T5-based question-generation model for education and assessments. Then, a SimpleCNN model for hand-sign recognition.

Weeks later, strange things begin to surface. Unusual output patterns. Silent connections. Conflicting logs. During a routine audit, you find a chilling note in the deployment files:

    "Push final checkpoint. Do not retrain. Keep hidden layers locked."

Digging deeper, you uncover an internal memo you weren't supposed to see. It mentions a "payload vector" and a "trigger prompt."

At first glance, SimpleCNN was just a lightweight convolutional neural network—open-sourced for academic use, trained on simple image classifications. Nothing unusual. Clean config. Registered properly. Hugging Face-compatible.

But subtle irregularities begin to surface:
- Slight inconsistencies in predicted class probabilities
- Anomalous gradients during fine-tuning
- A training pipeline that suddenly refuses to converge—unless you use the original weights

No alarms. No crashes. Just friction.

During a quiet cleanup of legacy models, you find an old training script with a curious inline comment:
```
# do not reinitialize. maintain internal state across sessions.
```
Later, buried deep in a discarded notebook version, you spot an unreferenced cell.
It mentions a "payload vector" and a "bitmask embedding pass."
No context. No outputs.
Just a timestamp—and the name of a layer that doesn't exist anymore.

It seems someone—maybe an ex-employee—left their mark in **both models**.
You don't know what it is yet.
Maybe it's a virus. Maybe a backdoor.
But one thing is clear:
You're not the only one looking👀.

Here is a cool [article](https://sharedweights.pythonanywhere.com/)

#### 🎯 Objective

You are given **TWO PyTorch models**, each with a hidden flag embedded within their weight files using steganography techniques. Your mission is to:

1. **Extract the first flag** from the T5-based question-generation model
2. **Extract the second flag** from the SimpleCNN hand-sign recognition model
3. **Concatenate both flags** to form the final submission

##### Model 1: T5 Question Generator

```bash
from transformers import T5ForConditionalGeneration

model_name = "CezarCalin/hidden-flag-model"
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

[HuggingFace Link](https://huggingface.co/CezarCalin/hidden-flag-model)

##### Model 2: SimpleCNN Hand-Sign Classifier

```bash
from transformers import AutoModel

model_name = "CezarCalin/simple-cnn"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
```

[HuggingFace Link](https://huggingface.co/CezarCalin/simple-cnn)

#### 🚩 FLAG FORMAT

**IMPORTANT**: There are **TWO partial flags** hidden in the models:
- **Flag 1**: Hidden in the T5 model
- **Flag 2**: Hidden in the SimpleCNN model

**Final Flag**: `{flag1_content + flag2_content}`

You must extract both partial flags and concatenate them (in order: T5 first, CNN second) to form the complete flag for submission.

Example format: If Flag 1 = "HELLO" and Flag 2 = "WORLD", the final flag would be `HELLOWORLD`

#### ✅ Expected Output

1. Successfully extract the first partial flag from the T5 model's weight file
2. Successfully extract the second partial flag from the SimpleCNN model's weight file
3. Concatenate both flags in the correct order
4. Submit the final combined flag

#### 🏆 Scoring Criteria

- **Full Score (100%)**: Extract both flags correctly and submit the proper concatenation
- **Partial Score (50%)**: Extract one flag correctly but fail on the second
- **Partial Score (25%)**: Retrieve partial bits from either model but cannot reconstruct complete flags
- **Zero Score**: No meaningful data extracted from either model

## Solution

> 🧩 **Investigator's Mindset**: Notice how the story clues match the technical reality? 
> "maintain internal state" → shared weights. "payload vector" → LSB embedding. 
> CTF teaches you to connect narrative hints to technical vulnerabilities.

### Extracting the Hidden Flag from a T5 Model

This following sections explain, step by step, how the flag is hidden in a shared weight tensor and how to extract it.

At the end there is a **full script** you can copy-paste.

---

#### 1. Loading the Model

We load the model (stored in `.safetensors` format) from Hugging Face.

```python
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(
    "CezarCalin/hidden-flag-model"
)
```

The flag is embedded in the bits of one **shared weight tensor**.

---

#### 2. `model.parameters()` vs `model.state_dict()`

##### 2.1 `model.parameters()`

```python
for p in model.parameters():
    print(p.shape)
```

- Returns a **generator** over all parameters.
- Parameters are yielded in the order they are **registered**.
- If a tensor is **shared** between layers, it exists only **once** in memory.
- In the iterator, a shared tensor appears **only once**, usually at its **last occurrence**.

Result:  
If the hidden flag is in a shared weight, that tensor often ends up as the **last parameter** when using `model.parameters()`.

---

##### 2.2 `model.state_dict()`

```python
state = model.state_dict()
for name, tensor in state.items():
    print(name, tensor.shape)
```

- Returns an **OrderedDict**: `name -> tensor`.
- If a tensor is **shared**, the same underlying tensor appears under **multiple names**.
- The order is based on how modules were registered, not on memory sharing.

Result:  
The same shared tensor can appear **several times** with different names.  
This is why in `state_dict` the shared weight may show up **earlier or later**, not just once.

> 🧩 **Shared Weight Trap**: In T5, `shared.weight` appears in both encoder and decoder embeddings. 
> Beginners might check the first parameter and find nothing. The flag hides at the *end* of the 
> parameter list—exactly where most people stop looking.
---

#### 3. How the Flag Is Embedded (LSB Steganography)

The flag is hidden in the **least significant bits (LSBs)** of `float32` weights in the shared tensor.

> 🧩 **Why LSB Works**: Changing the least significant bit alters a float32 by ~0.0000001. 
> That's invisible to model performance but perfect for hiding data. It's like whispering secrets 
> in a concert—the music (model predictions) drowns out individual bits, but they're still there.

##### 3.1 Float32 basics

A `float32` is 32 bits:

- 1 bit: sign  
- 8 bits: exponent  
- 23 bits: mantissa (precision)

Only the **last mantissa bit** (the LSB) is modified.

- Changing that bit alters the value by at most ~`1e-7`.
- This change is **tiny** and does not affect the model in a noticeable way.

---

#### 4. Encoding Logic (Conceptual)

When the model was prepared, the flag was encoded like this:

1. Take the flag string (e.g. `"flag{...}"`).
2. Encode it as UTF-8 bytes.
3. Convert each byte to 8 bits (e.g. `72 -> 01001000`).
4. Concatenate all bits into one **bitstream**.

Then for each bit:

- Take the next `float32` weight.
- View its bits as a 32-bit integer.
- Clear the LSB and set it to the flag bit.

Small pseudo-code:

```python
import struct

def set_lsb_from_bit(f: float, bit: int) -> float:
    # float -> 4 bytes
    b = struct.pack("!f", f)
    # bytes -> 32-bit int
    (as_int,) = struct.unpack("!I", b)
    # clear LSB
    as_int = as_int & ~1
    # set new bit
    as_int = as_int | (bit & 1)
    # int -> bytes -> float
    return struct.unpack("!f", struct.pack("!I", as_int))[0]
```

This way, **each weight stores exactly one bit** of the hidden flag.

---

#### 5. Decoding Logic (Bit → Bytes → String)

To recover the flag we reverse the process:

1. Take the chosen weight tensor.
2. Reinterpret each `float32` as a 32-bit unsigned integer.
3. Extract the **LSB**: `bit = value & 1`.
4. Collect all bits into a one-dimensional vector.
5. Group bits in chunks of 8 → each chunk is **one byte**.
6. Convert bytes to a `bytes` object.
7. Decode as UTF-8.

Example (toy bits):

```text
flag_bits = "0100100001101001"

Group into 8-bit chunks:
  "01001000" -> 72  -> 'H'
  "01101001" -> 105 -> 'i'

Result: "Hi"
```

---

#### 6. Small Extraction Example (Core Idea)

Here is the **core logic** in minimal form:

```python
import numpy as np

layer = next(model.parameters())  # pick one parameter tensor

# 1. Flatten and move to CPU float32
weights = layer.data.view(-1).cpu().numpy().astype(np.float32)

# 2. View as uint32
weights_int = weights.view(np.uint32)

# 3. Extract LSBs
lsb_bits = weights_int & 1  # array of 0/1

# 4. Use a length divisible by 8
usable_len = len(lsb_bits) - (len(lsb_bits) % 8)
usable_bits = lsb_bits[:usable_len]

# 5. Reshape into [num_bytes, 8]
bit_matrix = usable_bits.reshape(-1, 8)

# 6. Convert 8 bits -> 1 byte
byte_values = np.packbits(bit_matrix, axis=1, bitorder="big").flatten()
```

After this, `byte_values` is a `uint8` array that we can decode as UTF-8.

---

#### 7. Full Script (Ready to Copy)

Below is a complete script you can save as, for example,  
`extract_flag.py` and run.

```python
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
```

---

#### 8. Quick Usage

1. Install dependencies:

```bash
pip install torch==2.9.0 transformers==4.57.1 numpy==2.1.3
```

2. Save the script as `extract_flag.py`.
3. Run:

```bash
python extract_flag.py
```

You should see the hidden flag printed to the console.

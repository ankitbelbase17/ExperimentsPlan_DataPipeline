# Model Input Requirements - Current Status

## ✅ FULLY SIMPLIFIED MODELS (Person + Cloth Only)

### 1. CATVTON
**Model Signature:**
```python
def forward(self, person_img, garment_img, text_embeddings, timesteps, noise)
```

**Inputs:**
- ✅ `person_img` [B, 3, H, W] - Person image
- ✅ `garment_img` [B, 3, H, W] - Cloth image
- ✅ `text_embeddings` [B, 77, 768] - CLIP text embeddings (auto-generated from caption)
- ✅ `timesteps` [B] - Diffusion timestep
- ✅ `noise` [B, 4, h, w] - Gaussian noise

**Conditioning:** Cross-attention between garment features and text embeddings

---

### 2. IDM-VTON
**Model Signature:**
```python
def forward(self, person_img, garment_img, text_embeddings, timesteps, noise)
```

**Inputs:**
- ✅ `person_img` [B, 3, H, W]
- ✅ `garment_img` [B, 3, H, W]
- ✅ `text_embeddings` [B, 77, 768]
- ✅ `timesteps` [B]
- ✅ `noise` [B, 4, h, w]

**Conditioning:** Gated attention fusion for garment features

---

### 3. OOTDiffusion
**Model Signature:**
```python
def forward(self, person_img, garment_img, text_embeddings, timesteps, noise)
```

**Inputs:**
- ✅ `person_img` [B, 3, H, W]
- ✅ `garment_img` [B, 3, H, W]
- ✅ `text_embeddings` [B, 77, 768]
- ✅ `timesteps` [B]
- ✅ `noise` [B, 4, h, w]

**Conditioning:** Cross-attention conditioning

---

### 4. DiT (Diffusion Transformer)
**Model Signature:**
```python
def forward(self, x, t, y)
```

**Inputs:**
- ✅ `x` [B, 4, H/8, W/8] - Latent representation (encoded from person+cloth)
- ✅ `t` [B] - Timestep
- ✅ `y` [B] - Class label (0 for unconditional)

**Note:** DiT works at the latent level, so person+cloth are encoded together before passing to the model.

---

## ❌ NOT YET SIMPLIFIED (Still require additional inputs)

### 5. CP-VTON
**Current Model Signature:**
```python
def forward(self, person, garment, person_repr)
```

**Current Inputs:**
- ✅ `person` [B, 3, H, W] - Person image
- ✅ `garment` [B, 3, H, W] - Cloth image
- ❌ `person_repr` [B, 3, H, W] - **Person representation (pose/parsing) - STILL REQUIRED**

**Status:** ⚠️ Needs simplification to remove `person_repr`

---

### 6. VTON-GAN
**Current Model Signature (Generator):**
```python
def forward(self, person, garment, pose)
```

**Current Inputs:**
- ✅ `person` [B, 3, H, W] - Person image
- ✅ `garment` [B, 3, H, W] - Cloth image
- ❌ `pose` [B, 3, H, W] - **Pose map - STILL REQUIRED**

**Status:** ⚠️ Needs simplification to remove `pose`

---

## Summary Table

| Model | Person | Cloth | Text | Pose/Mask/Seg | Status |
|-------|--------|-------|------|---------------|--------|
| **CATVTON** | ✅ | ✅ | ✅ | ❌ None | ✅ Ready |
| **IDM-VTON** | ✅ | ✅ | ✅ | ❌ None | ✅ Ready |
| **OOTDiffusion** | ✅ | ✅ | ✅ | ❌ None | ✅ Ready |
| **DiT** | ✅ | ✅ | ✅ | ❌ None | ✅ Ready |
| **CP-VTON** | ✅ | ✅ | ❌ | ⚠️ person_repr | ❌ Needs work |
| **VTON-GAN** | ✅ | ✅ | ❌ | ⚠️ pose | ❌ Needs work |

---

## Recommendation

**For immediate use:** Focus on the 4 fully simplified models:
- CATVTON
- IDM-VTON
- OOTDiffusion
- DiT

These work perfectly with just the triplet dataset (initial_image, cloth_image, try_on_image).

**For CP-VTON and VTON-GAN:** These require architectural redesign to remove geometric dependencies. This is more complex because:
- CP-VTON uses explicit geometric warping (TPS)
- VTON-GAN uses pose for spatial alignment

Would you like me to simplify CP-VTON and VTON-GAN as well?

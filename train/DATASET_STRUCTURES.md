# Benchmark Dataset Directory Structures

This document describes the expected directory structure for each benchmark dataset.

## 🎯 Our Unified Structure (What Our Code Expects)

All benchmark datasets are converted to this **unified structure**:

```
{dataset_name}/test/
├── initial_image/          # Person/model images
│   ├── 000001_person.jpg
│   ├── 000002_person.jpg
│   └── ...
├── cloth_image/            # Garment images
│   ├── 000001_cloth_top.jpg
│   ├── 000002_cloth_dress.jpg
│   └── ...
└── try_on_image/           # Ground truth try-on results
    ├── 000001_vton.jpg
    ├── 000002_vton.jpg
    └── ...
```

**File Naming Convention:**
- Initial images: `{id}_person.{ext}`
- Cloth images: `{id}_cloth_{description}.{ext}`
- Try-on images: `{id}_vton.{ext}`

The `{id}` must match across all three directories for proper pairing.

---

## 📊 Benchmark-Specific Original Structures

### 1. VITON-HD

**Original Structure:**
```
VITON-HD/
└── test/
    ├── image/              # Person images (768×1024)
    │   ├── 000001_00.jpg
    │   ├── 000002_00.jpg
    │   └── ...
    ├── cloth/              # Garment images (768×1024)
    │   ├── 000001_1.jpg
    │   ├── 000002_1.jpg
    │   └── ...
    ├── image-densepose/    # DensePose maps (optional)
    ├── agnostic-v3.2/      # Agnostic person representation
    └── test_pairs.txt      # Pairing file
```

**test_pairs.txt format:**
```
000001_00.jpg 000001_1.jpg
000002_00.jpg 000002_1.jpg
...
```

**Conversion to Our Structure:**
```bash
# Create unified structure
mkdir -p baselines/viton-hd/test/{initial_image,cloth_image,try_on_image}

# Copy and rename
cd VITON-HD/test
for pair in $(cat test_pairs.txt); do
    person=$(echo $pair | awk '{print $1}')
    cloth=$(echo $pair | awk '{print $2}')
    id=$(echo $person | cut -d'_' -f1)
    
    cp image/$person baselines/viton-hd/test/initial_image/${id}_person.jpg
    cp cloth/$cloth baselines/viton-hd/test/cloth_image/${id}_cloth.jpg
    # Note: VITON-HD doesn't provide ground truth, use person image as placeholder
    cp image/$person baselines/viton-hd/test/try_on_image/${id}_vton.jpg
done
```

---

### 2. DressCode

**Original Structure:**
```
DressCode/
├── upper_body/
│   └── test/
│       ├── images/         # Person images
│       ├── clothes/        # Garment images
│       └── test_pairs.txt
├── lower_body/
│   └── test/
│       ├── images/
│       ├── clothes/
│       └── test_pairs.txt
└── dresses/
    └── test/
        ├── images/
        ├── clothes/
        └── test_pairs.txt
```

**test_pairs.txt format:**
```
013563_0.jpg 013563_1.jpg
013564_0.jpg 013564_1.jpg
...
```

**Conversion to Our Structure:**
```bash
# For each category (upper_body, lower_body, dresses)
mkdir -p baselines/dresscode/test/{initial_image,cloth_image,try_on_image}

# Combine all categories
for category in upper_body lower_body dresses; do
    cd DressCode/$category/test
    
    for pair in $(cat test_pairs.txt); do
        person=$(echo $pair | awk '{print $1}')
        cloth=$(echo $pair | awk '{print $2}')
        id="${category}_$(echo $person | cut -d'_' -f1)"
        
        cp images/$person baselines/dresscode/test/initial_image/${id}_person.jpg
        cp clothes/$cloth baselines/dresscode/test/cloth_image/${id}_cloth.jpg
        cp images/$person baselines/dresscode/test/try_on_image/${id}_vton.jpg
    done
done
```

---

### 3. DeepFashion

**Original Structure:**
```
DeepFashion/
└── In-shop_Clothes_Retrieval/
    ├── img/
    │   ├── MEN/
    │   │   ├── Denim/
    │   │   ├── Jackets_Vests/
    │   │   └── ...
    │   └── WOMEN/
    │       ├── Blouses_Shirts/
    │       ├── Dresses/
    │       └── ...
    └── list_eval_partition.txt
```

**list_eval_partition.txt format:**
```
img/MEN/Denim/id_00000001/01_1_front.jpg test
img/MEN/Denim/id_00000001/01_2_side.jpg test
...
```

**Conversion to Our Structure:**
```bash
mkdir -p baselines/deepfashion/test/{initial_image,cloth_image,try_on_image}

# Extract test set
grep " test" DeepFashion/In-shop_Clothes_Retrieval/list_eval_partition.txt | \
while read line; do
    path=$(echo $line | awk '{print $1}')
    
    # Extract ID and view
    id=$(echo $path | grep -oP 'id_\d+')
    view=$(echo $path | grep -oP '\d+_\d+_\w+' | tail -1)
    
    # Use front view as person, other views as cloth
    if [[ $view == *"front"* ]]; then
        cp DeepFashion/In-shop_Clothes_Retrieval/$path \
           baselines/deepfashion/test/initial_image/${id}_person.jpg
        cp DeepFashion/In-shop_Clothes_Retrieval/$path \
           baselines/deepfashion/test/try_on_image/${id}_vton.jpg
    else
        cp DeepFashion/In-shop_Clothes_Retrieval/$path \
           baselines/deepfashion/test/cloth_image/${id}_cloth_${view}.jpg
    fi
done
```

---

### 4. Our Custom Test Set (dataset_ultimate)

**Structure:**
```
dataset_ultimate/
└── test/
    ├── easy/
    │   ├── female/
    │   │   ├── initial_image/
    │   │   ├── cloth_image/
    │   │   └── try_on_image/
    │   └── male/
    │       ├── initial_image/
    │       ├── cloth_image/
    │       └── try_on_image/
    ├── medium/
    │   └── ...
    └── hard/
        └── ...
```

**Already in unified format!** No conversion needed.

---

## 🔄 Automated Conversion Scripts

### VITON-HD Converter
```bash
#!/bin/bash
# convert_vitonhd.sh

SOURCE_DIR="path/to/VITON-HD/test"
TARGET_DIR="baselines/viton-hd/test"

mkdir -p "$TARGET_DIR"/{initial_image,cloth_image,try_on_image}

while IFS=' ' read -r person cloth; do
    id=$(echo "$person" | cut -d'_' -f1)
    
    cp "$SOURCE_DIR/image/$person" "$TARGET_DIR/initial_image/${id}_person.jpg"
    cp "$SOURCE_DIR/cloth/$cloth" "$TARGET_DIR/cloth_image/${id}_cloth.jpg"
    cp "$SOURCE_DIR/image/$person" "$TARGET_DIR/try_on_image/${id}_vton.jpg"
done < "$SOURCE_DIR/test_pairs.txt"

echo "VITON-HD conversion complete!"
```

### DressCode Converter
```bash
#!/bin/bash
# convert_dresscode.sh

SOURCE_DIR="path/to/DressCode"
TARGET_DIR="baselines/dresscode/test"

mkdir -p "$TARGET_DIR"/{initial_image,cloth_image,try_on_image}

for category in upper_body lower_body dresses; do
    while IFS=' ' read -r person cloth; do
        id="${category}_$(echo "$person" | cut -d'_' -f1)"
        
        cp "$SOURCE_DIR/$category/test/images/$person" \
           "$TARGET_DIR/initial_image/${id}_person.jpg"
        cp "$SOURCE_DIR/$category/test/clothes/$cloth" \
           "$TARGET_DIR/cloth_image/${id}_cloth.jpg"
        cp "$SOURCE_DIR/$category/test/images/$person" \
           "$TARGET_DIR/try_on_image/${id}_vton.jpg"
    done < "$SOURCE_DIR/$category/test/test_pairs.txt"
done

echo "DressCode conversion complete!"
```

---

## 📝 Summary

### Quick Reference Table

| Dataset | Original Format | Conversion Needed | Ground Truth Available |
|---------|----------------|-------------------|----------------------|
| **VITON-HD** | Separate dirs + pairs.txt | ✅ Yes | ❌ No (use person as GT) |
| **DressCode** | Category-based + pairs.txt | ✅ Yes | ❌ No (use person as GT) |
| **DeepFashion** | Hierarchical structure | ✅ Yes | ❌ No (use front view as GT) |
| **Ours** | Unified structure | ❌ No | ✅ Yes |

### Important Notes

1. **Ground Truth Limitation**: Most benchmarks don't provide actual try-on ground truth. We use the original person image as a reference.

2. **File Naming**: The `{id}` prefix is crucial for matching person-cloth-tryon triplets.

3. **Image Sizes**:
   - VITON-HD: 768×1024
   - DressCode: 1024×768
   - DeepFashion: Variable
   - Ours: 512×512 (configurable)

4. **Automatic Resizing**: Our evaluation code automatically resizes all images to 512×512 for consistency.

---

## 🚀 Quick Setup

1. **Download benchmark datasets** from official sources
2. **Run conversion scripts** (provided above)
3. **Verify structure**:
   ```bash
   ls baselines/viton-hd/test/
   # Should show: initial_image/ cloth_image/ try_on_image/
   ```
4. **Run evaluation**:
   ```bash
   bash bash_scripts/metrics_vitonhd.sh
   ```

---

**Last Updated**: 2026-01-22

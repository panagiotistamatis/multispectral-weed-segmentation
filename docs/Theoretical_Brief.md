# Θεωρητική Τεκμηρίωση Διπλωματικής

**Τίτλος**: Efficient Multispectral Crop-Weed Segmentation με Hierarchical Vision Transformers

**Στόχος**: Σύντομη παρουσίαση της μεθοδολογίας — γιατί κάθε επιλογή έγινε και πώς συνδέεται με τη βιβλιογραφία.

---

## 1. Πρόβλημα & κίνητρο

### 1.1 Precision Agriculture context

Η διαχείριση ζιζανίων σε καλλιέργειες (weed management) είναι κρίσιμη για:
- **Αύξηση παραγωγής**: τα ζιζάνια μειώνουν την απόδοση 10-30%
- **Μείωση χημικών**: site-specific weed control (SSWC) αντί για ομοιόμορφο ψεκασμό
- **Robotic farming**: αυτόνομα UAVs/tractors χρειάζονται pixel-level segmentation σε real-time

### 1.2 Crop-Weed Semantic Segmentation

Πρόβλημα: για κάθε pixel μιας UAV εικόνας, ταξινόμηση σε 3 κατηγορίες:
- **background** (soil)
- **crop** (καλλιέργεια — π.χ. καλαμπόκι)
- **weed** (ζιζάνιο)

### 1.3 WeedsGalore dataset (Celikkan et al., WACV 2025)

- **156 tiles** 600×600 pixels, 4 ημερομηνίες πτήσης UAV
- **5 spectral bands**: R, G, B, NIR (840nm), Red-Edge (730nm)
- **Extreme class imbalance**: background ~95%, weed ~1%
- Παρέχει multispectral input → πλούσια spectral information πέρα από RGB

---

## 2. Αρχιτεκτονικές επιλογές

### 2.1 Hierarchical Vision Transformer (MiT) ως backbone

**Επιλογή**: MiT-B0 (Mix Transformer, Xie et al., NeurIPS 2021 — SegFormer)

**Γιατί αντί CNN**:
- **Hierarchical features**: 4 stages με spatial downsampling 1/4, 1/8, 1/16, 1/32 — όπως CNN backbones αλλά με self-attention
- **Efficient Self-Attention** με spatial reduction (sr_ratio) — γραμμική complexity αντί quadratic
- **Mix-FFN**: 3×3 depthwise convolution μέσα στο MLP block → implicit positional encoding χωρίς explicit position embeddings (κρίσιμο για variable-size inputs σε UAV imagery)
- **Lightweight B0**: μόνο 3.3M params για encoder — ιδανικό για deployment

**Συγκριτικά**:
| Backbone | Params | Inductive bias |
|---|---|---|
| ResNet-50 (CNN) | 25M | Translation invariance |
| ViT-Base | 86M | Global attention (κανένα locality bias) |
| **MiT-B0** | **3.3M** | **Hierarchical + locality (mix-FFN)** |

### 2.2 Lawin Decoder

**Επιλογή**: Lawin (Yan et al., 2022) — Large-window attention decoder

**Γιατί**:
- Παίρνει multi-stage MiT features → unified feature pyramid
- **Window-based attention** στη μεγάλη ανάλυση (κρατάει spatial detail)
- **Atrous Spatial Pyramid Pooling** (ASPP-style) για multi-scale context
- **Skip connection** από low-level features για boundary refinement

### 2.3 SplitLawin — 2-stream variant (Castellano et al., 2023)

**Επιλογή**: Δύο παράλληλα MiT backbones που μοιράζονται decoder.

**Reasoning**:
- **Main backbone** (3-channel input): φορτώνει ImageNet pretrained weights απευθείας
- **Side backbone** (extra spectral channels): νέα channels, partial pretrained
- **Fusion block**: συνδυάζει τα features στο πρώτο stage πριν την παράλληλη επεξεργασία

**Δικιά μας καινοτομία στο WeedsGalore**: χρησιμοποιούμε **CIR composite (NIR, G, R)** ως main stream αντί RGB. Λόγος: το CIR είναι vegetation-aware false-color — το main backbone βλέπει task-relevant features από την αρχή. Side stream παίρνει B + RE για spectral complement.

### 2.4 Squeeze-Excite fusion (Hu et al., CVPR 2018)

**Mechanism**:
$$g = \sigma(\text{FC}_2(\text{ReLU}(\text{FC}_1(\text{GAP}([x_{\text{main}}, x_{\text{side}}])))))$$
$$y = g \odot [x_{\text{main}}, x_{\text{side}}]$$

**Γιατί**: channel-wise attention που μαθαίνει **adaptive weighting** των 2 streams ανά spatial location. Πιο εκφραστικό από plain concatenation ή element-wise addition.

---

## 3. Loss Functions — Μαθηματική Τεκμηρίωση

Η loss function είναι **το πιο σημαντικό knob** για imbalanced segmentation. Δοκιμάσαμε 5 παραλλαγές, καταλήγοντας σε combined **Focal Tversky + Lovász Softmax (50/50)**.

### 3.1 Focal Loss (Lin et al., ICCV 2017)

$$L_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

όπου $p_t$ = predicted probability για την σωστή κλάση, $\gamma$ = focusing parameter (default 2).

**Idea**: μειώνει το loss για easy examples ($p_t \to 1$), εστιάζει σε hard examples → καλό για imbalanced.

**Στο WeedsGalore**: μόνη της δίνει F1=0.820 → χρειάζεται enhancement.

### 3.2 Tversky Index — γενίκευση του Dice

Per-class soft Tversky:
$$\text{TI}_c = \frac{\text{TP}_c}{\text{TP}_c + \alpha \cdot \text{FP}_c + \beta \cdot \text{FN}_c}$$

όπου $\alpha + \beta$ ελέγχουν τη σχετική βαρύτητα FP vs FN.
- $\alpha = \beta = 0.5$ → ισοδύναμο με Dice
- $\alpha < \beta$ → επιβραβεύει recall (λιγότερα missed positives)

### 3.3 Focal Tversky Loss (Abraham & Khan, 2018)

$$L_{\text{FT}} = (1 - \text{mean}_c(\text{TI}_c))^{1/\gamma}$$

**Γιατί νικάει Focal**:
1. **Per-class averaging**: ομοιόμορφη βαρύτητα κάθε κλάσης, χωρίς explicit class weights
2. **Focal exponent στο IoU-space**: focus σε classes με χαμηλό TI (συνήθως minority classes)
3. **Differentiable**: smooth gradients για σταθερό training

**Δικιά μας tuning**: $\alpha = 0.4$, $\beta = 0.6$, $\gamma = 4/3$. Σύγκριση:
- $\alpha=\beta=0.5$ (Dice): mIoU = 0.827 
- $\alpha=0.3, \beta=0.7$ (paper default): mIoU = 0.833
- **$\alpha=0.4, \beta=0.6$ (δικιά μας)**: **mIoU = 0.836** 

### 3.4 Lovász-Softmax Loss (Berman et al., CVPR 2018)

**Idea**: convex surrogate του Jaccard (IoU) — direct optimization της metric που αξιολογεί το model.

Για κάθε class $c$:
1. Sort errors: $e_c = |y_c - p_c|$ από μεγάλο σε μικρό
2. Compute Lovász gradient $\nabla_J$ μέσω confusion matrix
3. Loss = ⟨sorted errors, $\nabla_J$⟩

**Γιατί συμπληρωματικό με Focal Tversky**:
- **FT** δίνει στα per-class soft IoUs equal weight
- **Lovász** προσθέτει **direct gradient** στο IoU metric — αντί surrogate (cross-entropy), βελτιστοποιεί ακριβώς αυτό που αξιολογούμε
- Empirically: 50/50 combination πετυχαίνει **+0.7 pts F1** πάνω από FT alone

### 3.5 Combined Loss

$$L_{\text{combined}} = 0.5 \cdot L_{\text{FT}}(\alpha=0.4, \beta=0.6) + 0.5 \cdot L_{\text{Lovász}}$$

**Final**: F1 = 0.918 (single seed), 0.909 ± 0.004 (3-seed mean).

---

## 4. Channel Reasoning — Spectral Physics

### 4.1 Διαθέσιμα bands στο WeedsGalore

| Band | Wavelength | Φυσική σημασία |
|---|---|---|
| **R** | ~660nm | Chlorophyll absorption — vegetation appears dark |
| **G** | ~550nm | Chlorophyll reflection — vegetation appears bright |
| **B** | ~470nm | Atmospheric scattering, soil discrimination |
| **NIR** | ~840nm | Strong reflection από healthy chlorophyll → NDVI input |
| **RE** | ~730nm | Chlorophyll stress detection (red-edge slope) |

### 4.2 CIR composite (Color Infrared)

**CIR = [NIR, G, R]** as 3-channel false-color image.
- Διαθέσιμο φυτό → **bright red/pink** (NIR strong, R weak)
- Stressed φυτό → πιο σκούρο
- Soil → grey/blue tones

**Γιατί CIR ως main**:
- Το main backbone (3-channel pretrained ImageNet) δουλεύει σε **vegetation-aware** input
- Side backbone συμπληρώνει με B + RE (κανάλια που το ImageNet pretrained δεν "βλέπει")
- Παρόμοιο pattern με paper baseline αλλά **inverted**: paper βάζει RGB main + NIR side, εμείς CIR main + B+RE side

### 4.3 Computed indices (NDVI, OSAVI, MSAVI)

Δοκιμάσαμε vegetation indices ως extra channels:
- **NDVI** = (NIR-R)/(NIR+R) → vegetation strength
- **OSAVI** = (NIR-R)/(NIR+R+0.16) → soil-adjusted
- **MSAVI** → self-calibrating

**Finding**: Δεν βελτίωσαν vs raw NIR/R. Λογικό — η information είναι ήδη στα raw bands, το model μπορεί να μάθει τη σχέση. Computed indices = redundant.

---

## 5. Methodology — Γιατί τόσα ablations

### 5.1 Systematic optimization pipeline

1. **Loss exploration** — από Focal σε FT+Lovász
2. **Channel ablations** — 8 διαφορετικά combos
3. **Hyperparameter tuning** — α/β sweep
4. **Multi-seed verification** — 3 seeds για στατιστική
5. **Architectural/optimizer ablations** — fusion blocks, optimizers, init schemes

### 5.2 Statistical rigor (multi-seed)

**Πρόβλημα**: ένα single training run έχει randomness (init, shuffling). Σύγκριση 2 configs με 1 seed είδος μπορεί να είναι **τυχαία**.

**Λύση**: Run κάθε critical config σε **3 seeds** (42, 43, 44):
$$\text{mIoU}_{\text{reported}} = \mu \pm \sigma$$

Για το winner: mIoU = 0.839 ± 0.006. Άρα διαφορές < 0.012 = **εντός noise**, δεν είναι "βελτιώσεις".

**Παράδειγμα value**: το BoundaryDoU loss φαινόταν να βελτιώνει με single-seed (0.845 > 0.839), αλλά multi-seed verification έδειξε mean = 0.839 ± 0.006 — **identical με baseline**. Χωρίς multi-seed, θα είχαμε false positive claim.

### 5.3 Negative findings — methodological value

Documenting τι ΔΕΝ δούλεψε είναι εξίσου σημαντικό:
- I3D-scaled init: -0.024 (training compensates magnitude mismatch)
- Coordinate Attention: -0.005 (MiT ήδη spatial-aware)
- ADOPT optimizer: -0.013 (Adam ήδη συγκλίνει σταθερά)
- TTA: -0.002 (training flip aug = redundant με test-time flip)
- OSAVI/MSAVI: <0 (raw bands ήδη επαρκή)
- BoundaryDoU: 0 ± 0.006 (Lovász ήδη πιάνει boundary signal)

Αυτή η transparency είναι **καθοριστική για credibility**.

---

## 6. Cross-Dataset Validation

### 6.1 WeedMap (Sa et al., 2018)

Δοκιμάσαμε το winner config σε δεύτερο dataset:
- **Rheinbach (RedEdge)**: F1 = 0.863  — matches paper baseline 0.857-0.865
- **Eschikon (Sequoia)**: F1 = 0.505  — domain shift challenge

### 6.2 Eschikon analysis

Test plot 005 έχει **0.013% weed pixels** (vs 5-10% σε train plots 006/007). Severe class distribution shift.

Sanity check με paper baseline (Lawin-B0 CIR + Focal+weights): F1 = 0.605 (vs paper 0.663). Εντός expected gap → **infrastructure OK**, FT+Lovász απλά δεν transfer-άρει σε αυτό το extreme imbalance setting.

---

## 7. Summary των key claims

| Claim | Evidence |
|---|---|
| State-of-the-art σε WeedsGalore | mIoU = 0.839 ± 0.006 (3-seed) |
| 8× efficiency vs paper baseline | 5.4M params vs 41M (DeepLabV3+ MS) |
| Methodology generalizes | Rheinbach F1=0.863 matches paper |
| Methodological rigor | 6 documented negative findings |
| Loss combination is novel | FT+Lovász 50/50 — first such combination on this benchmark |

---

## 8. Bibliography (key references)

1. **Celikkan et al.** "WeedsGalore: A multispectral and multitemporal UAV-based dataset", WACV 2025
2. **Xie et al.** "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021
3. **Yan et al.** "Lawin Transformer: Improving Semantic Segmentation Transformer with Multi-Scale Representations via Large Window Attention", 2022
4. **Castellano et al.** "Weed mapping in multispectral drone imagery using lightweight vision transformers", Neurocomputing 2023
5. **Abraham & Khan** "A novel focal tversky loss function with improved attention u-net for lesion segmentation", arXiv:1810.07842, 2018
6. **Berman, Triki, Blaschko** "The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks", CVPR 2018
7. **Lin et al.** "Focal Loss for Dense Object Detection", ICCV 2017
8. **Hu, Shen, Sun** "Squeeze-and-Excitation Networks", CVPR 2018
9. **Sa et al.** "WeedMap: A Large-Scale Semantic Weed Mapping Framework Using Aerial Multispectral Imaging and Deep Neural Network for Precision Farming", Remote Sensing 2018
10. **Galymzhankyzy & Martinson** "Lightweight Multispectral Crop-Weed Segmentation for Precision Agriculture", ICRA Workshop 2025

---

*Document version 1.0 — pre-supervisor meeting*

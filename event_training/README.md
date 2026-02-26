Use TSM integrated at the feature-map level with a 2D ResNet backbone that processes frames independently, then applies a lightweight Temporal Shift Module on the per-frame feature maps. This gives near‑3D temporal modeling at very low CPU cost, reuses ImageNet weights, and easy to implement and optimize for CPU-only hardware.

CPU OPTIMIZATION & DEPLOYMENT ADVICE
- Quantize after training (post-training static quantization) - reduces latency; validate accuracy
- Export to TorchScript (torch.jit.trace) and benchmark; if faster, use for inference
- ONNX -> ONNX Runtime / OpenVINO yields large CPU speedups; export and test
- Cache per-frame backbone features if classifier is run on overlapping windows; trades storage for speed
- Set OMP_NUM_THREADS / MKL_NUM_THREADS to avoid thread oversubscription when using Dataloader + Pytorch

EVAL + DIAGNOSTICS TO RUN EARLY (classifier)
- Per-class confusion matrix - which events are confused
- Temporal +-k accuracy (k=1, 3, 5 frames) using clip center as anchor
- Detector -> Classifier pipeline test: run detector on videos, extract candidates, run classifier, compare predicted event frame (clip center) vs ground truth
- Ablation: compare TSM vs temporal average pooling to quantify TSM benefit

STRATEGIES FOR OVERLAPPING EVENTS (within a clip)
- Balanced sampling: batches should include clips that have single events, multiple events, and pure background
- Loss weighting: use per-class weights or focal loss to handle class imbalance and sparse positives (likely given total videos)
- Hard negative mining: after initial detector training, run detector in training videos, collect false positives and include them as negative examples for classifier heatmap training - increases accuracy
- Data augmentation: temporal jitter (shift clip center by +-1-3 frames) so classifier learns to localize when the detector is off by a few frames (robust)

EVAL METRICS (detector)
- Event detection recall at IoU or +-k frames
- Per-class precision/recall/F1 - classification of localized events
- Mean aboslute frame error between predicted and ground truth
- False positives per minute - (full pipeline)
- Sequence confusion: how often predicted order of events matches ground truth (some swallows start before others finish, may cause confusion)

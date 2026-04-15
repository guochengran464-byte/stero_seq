# stero_seq
HCC spatial transcriptomics analysis pipeline (BGI Stereo-seq)

## Pipeline
- step1_Segmentation: cell segmentation from ssDNA images
- step2_SpatialObject_SCT: Seurat SCT clustering (bin50 + cellbin)
- step3: spatial constrained clustering (Leiden)
- step4: RCTD deconvolution

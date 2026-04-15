########
# modified: median --> mean

LoadBGI_SpatialATAC <- function(id,
                                filename,
                                outdir = getwd(),
                                bin_data = T,
                                bin_size = 50,
                                cell_mask = F,
                                # area_mask = F,
                                save_as = "rds",
                                pro_name = "Spatial",
                                UMI_GreyScale_Image = T,
                                assays = "SpatialATAC",
                                # slice = "slice1",
                                nThread=16,
                                tabix_bin = '/share/app/htslib/bin/tabix',
                                bgzip_bin = '/share/app/htslib/bin/bgzip',
                                gtf_path,
                                delete_bg = T,csv=F) {
    if (csv)
	{dat<-read.csv(file=filename)
  print('mark')}
    else
    {dat <- data.table::fread(file = filename, nThread = 8)}
    colnames(dat)[1:7] <- c("chr", "start", "end", "cellid", "UMICount", "x", "y")
    colnames(dat)[1:8] <- c("geneID", "chr", "start", "end", "cellid", "UMICount", "x", "y")
    if (bin_data) {
        rowname_style <- paste0(id, "_BIN")

        dat$x <- trunc((dat$x - min(dat$x)) / bin_size + 1)
        dat$y <- trunc((dat$y - min(dat$y)) / bin_size + 1)

        if (area_mask) {
            dat$bin_ID <- max(dat$x) * (dat$y - 1) + dat$x
            Area_ <- dat$AreaID[!duplicated(dat$bin_ID)]
        }

        if ("MIDCounts" %in% colnames(dat)) {
            dat <- dat[, sum(MIDCounts), by = .(geneID, x, y)]
        } else {
            dat <- dat[, sum(UMICount), by = .(geneID, x, y)]
        }
        dat$bin_ID <- max(dat$x) * (dat$y - 1) + dat$x
        bin.coor <- dat[, sum(V1), by = .(x, y)]
        if (UMI_GreyScale_Image) {
            scale_grey <- bin.coor$V1 / quantile(bin.coor$V1, probs = 0.95)
            scale_grey[scale_grey > 1] <- 1
            tissue_lowres_image <- as.matrix(Matrix::sparseMatrix(
            i = bin.coor$y,
            j = bin.coor$x,
            x = scale_grey
            ))
            tissue_lowres_image_r <- raster::raster(tissue_lowres_image)
            tissue_lowres_image_r <- raster::writeRaster(tissue_lowres_image_r, file.path(outdir, paste0(pro_name, "_UMIGreyScaleFakeIMG.tiff")), overwrite=T)
        } else tissue_lowres_image <- matrix(0, max(bin.coor$y), max(bin.coor$x))
    }

    if (cell_mask) {
        rowname_style <- paste0(id, "_CELL")
        
        colnames(dat)[1:9] <- c("geneID", "chr", "start", "end", "cellid", "UMICount", "x", "y","label")
        if (delete_bg){
	dat<-subset(dat,label>0)}
        dat$x <- dat$x - min(dat$x) + 1
        dat$y <- dat$y - min(dat$y) + 1

        dat_bkp <- dat
        dat_bkp <- dat_bkp[, sum(UMICount), by = .(geneID, x, y)]
        bin.coor.bkp <- dat_bkp[, sum(V1), by = .(x, y)]

        if (UMI_GreyScale_Image) {
            scale_grey <- bin.coor.bkp$V1 / quantile(bin.coor.bkp$V1, probs = 0.95)
            scale_grey[scale_grey > 1] <- 1
            tissue_lowres_image <- as.matrix(Matrix::sparseMatrix(
                i = bin.coor.bkp$y,
                j = bin.coor.bkp$x,
                x = scale_grey
            ))
            tissue_lowres_image_r <- raster::raster(tissue_lowres_image)
            tissue_lowres_image_r <- raster::writeRaster(tissue_lowres_image_r, file.path(outdir, paste0(pro_name, "_UMIGreyScaleFakeIMG.tiff")), overwrite=T)
        } else tissue_lowres_image <- matrix(1, max(bin.coor.bkp$y), max(bin.coor.bkp$x))

        dat <- dat[dat$label != 0,]
        dat.x <- as.data.frame(dat[, ceiling(mean(x)), by = .(label)])
        hash.x <- data.frame(row.names = dat.x$label, values = dat.x$V1)
        dat.y <- as.data.frame(dat[, ceiling(mean(y)), by = .(label)])
        hash.y <- data.frame(row.names = dat.y$label, values = dat.y$V1)
        dat$x <- hash.x[sprintf("%d", dat$label), "values"]
        dat$y <- hash.y[sprintf("%d", dat$label), "values"]

        te <- data.frame(unique(paste(dat$x, dat$y, dat$label, sep = "_"))) # slow
        colnames(te) <- "xyb"
        split_b <- stringr::str_split(te$xyb, "_")
        te$x <- as.vector(sapply(split_b, "[", 1))
        te$y <- as.vector(sapply(split_b, "[", 2))
        te$b <- as.vector(sapply(split_b, "[", 3))
        te$xy <- paste(te$x, te$y, sep = "_")
        dp <- which(duplicated(te$xy))
        dplen <- length(dp)

        for (i in dplen) {
            tmp <- dp[i]
            xy <- te[tmp, ]$xy
            cxy <- which(te$xy %in% xy)
            cxylen <- length(cxy)
            keep <- te[cxy[1], ]$b
            for (m in 2:cxylen) {
                c <- which(dat$label %in% te[cxy[m], ]$b)
                dat$label[c] <- as.numeric(keep)
            }
        }
        dat$label <- as.numeric(dat$label)
        dat <- dat[, sum(UMICount), by = .(geneID, x, y,label)]
        dat$bin_ID <- dat$label
        bin.coor <- dat[, sum(V1), by = .(x, y)]
    }
    
    export_dat <- dat[, .(geneID, bin_ID, V1)]
    export_dat[, c("chr", "start", "end") := tstrsplit(geneID, "-", fixed = TRUE)]
    export_dat[, cellID := paste0(rowname_style, ".", bin_ID)]
    export_dat[, `:=`(start = as.numeric(start), end = as.numeric(end))]
    if (!is.null(gtf_path)){
        message("Extracting chromosome order from GTF...")
        gtf_chr_order <- system(paste0("zgrep -v '^#' ", gtf_path, " | cut -f1 | uniq"), intern = TRUE)
        current_chrs <- unique(export_dat$chr)
        final_levels <- c(
            intersect(gtf_chr_order, current_chrs), 
            setdiff(current_chrs, gtf_chr_order)
        )
        export_dat[, chr := factor(chr, levels = final_levels)]
    }
    setorder(export_dat, chr, start)
    final_fragment <- export_dat[, .(chr = as.character(chr), start, end, cellID, V1)]
    
    if (bin_data) {
        output_tsv <- file.path(outdir, paste0(pro_name, '_Bin', bin_size, "_fragments.tsv"))
    }
    if (cell_mask) {
        output_tsv <- file.path(outdir, paste0(pro_name, "_CellBin_fragments.tsv"))
    }
    fwrite(
        final_fragment, 
        file = output_tsv, 
        sep = "\t", nThread = nThread,
        col.names = FALSE, 
        scipen = 999
    )
    message("Compressing with bgzip...")
    system(paste(bgzip_bin, "-f", output_tsv)) # 生成 .tsv.gz

    final_gz <- paste0(output_tsv, ".gz")
    message("Creating tabix index...")
    tabix_cmd <- paste(tabix_bin, "-p bed -f", final_gz)
    system(tabix_cmd)
    message("Process finished. File: ", final_gz)
    
    rm(export_dat)
    gc()
    # if (area_mask) {
    #     hash.A <- data.frame(row.names = paste(rowname_style, rownames(hash.B), sep = '.'), area = Area_)
    #     object <- Seurat::CreateSeuratObject(mat, project = assay, assay = assay, meta.data = hash.A)
    # } else object <- Seurat::CreateSeuratObject(mat, project = assay, assay = assay)
    
    cell = unique(final_fragment$cellID)
    frags <- CreateFragmentObject(path = final_gz, cells = as.character(cell))
    peaks <- CallPeaks(frags)
    counts <- FeatureMatrix(fragments = frags,features = peaks,cells = as.character(cell))
    assay <- CreateChromatinAssay(counts, fragments = frags)
    object <- CreateSeuratObject(assay, assay = assays)
    
    tissue_positions <- distinct(dat[, .(x, y, bin_ID)])
    coord <- data.frame(
        row.names = paste(rowname_style, tissue_positions$bin_ID, sep = "."), # tissue = 1,
        x = tissue_positions$x, y = tissue_positions$y
    )
    colnames(coord) = c('Spatial_1','Spatial_2')
    coord <- coord[Cells(object), ]
    
    object[['Spatial']] <- SeuratObject::CreateDimReducObject(embeddings = as.matrix(coord), key = 'Spatial_', assay = assays)
    object <- AddMetaData(object, metadata = coord) 
    Seurat::DefaultAssay(object) <- assays

    # object <- subset(object, subset = nCount_Spatial > 0)

    if (save_as == "rds") {
        if (bin_data) {
            saveRDS(object, file = file.path(outdir, paste0(pro_name, '_Bin', bin_size, ".rds")))
        }
        if (cell_mask) {
            saveRDS(object, file = file.path(outdir, paste0(pro_name, "_CellBin.rds")))
        }
    }
    if (save_as == "h5ad") {
        genes <- as.data.frame(rownames(object), row.names = rownames(object))
        names(genes) <- "Gene"

        cells <- object@meta.data

        coordinates <- list(matrix(coord))
        names(coordinates) <- "spatial"

        ad <- anndata::AnnData(X = object@assays$Spatial@counts, obs = genes, var = cells, varm = coordinates)
        ad <- ad$T

        if (bin_data) {
            ad$write_h5ad(file.path(outdir, paste0(pro_name, '_Bin', bin_size, , ".h5ad")))
        }
        if (cell_mask) {
            ad$write_h5ad(file.path(outdir, paste0(pro_name, "_CellBin.h5ad")))
        }
    }
                            
    supported_format <- c("rds", "h5ad")
    if (!save_as %in% supported_format) {
        write("Warning: not saving to file or file format (specified in save_as) not supported, supported format: rds, h5ad", stderr())
    }

    return(object)

    ##bin data
    # out <- as.data.frame(dat)
    # out <- cbind(dat$y,dat$x,out)
    # colnames(out)[1:2] <- c(paste0(rowname_style,bs,".y"),paste0(rowname_style,bs,".x"))
    # fwrite(out,paste0(pro_name,"_bin",bs,"_information.txt"),col.names=T,row.names=F,sep="\t",quote=F)
    # out <- as.data.frame(cbind(paste0(rowname_style,unique(dat$bin_ID)),bin.coor$y,bin.coor$x))
    # colnames(out) <- c(paste0(rowname_style,bs),paste0(rowname_style,bs,".y"),paste0(rowname_style,bs,".x"))
    # rownames(out) <- out[,1]
    # fwrite(out,paste0(pro_name,"_bin",bs,"_position.txt"),col.names=T,row.names=F,sep="\t",quote=F)
    # bkpos <- out

    ##cell mask
    # out <- as.data.frame(dat)
    # out <- cbind(dat$y,dat$x,out)
    # colnames(out)[1:2] <- c(paste0(rowname_style,".y"),paste0(rowname_style,".x"))
    # fwrite(out,paste0(pro_name,"_cell","_information.txt"),col.names=T,row.names=F,sep="\t",quote=F)
    # out <- as.data.frame(cbind(paste0(rowname_style,unique(dat$bin_ID)),bin.coor$y,bin.coor$x))
    # colnames(out) <- c(paste0(rowname_style,bs),paste0(rowname_style,bs,".y"),paste0(rowname_style,bs,".x"))
    # rownames(out) <- out[,1]
    # fwrite(out,paste0(pro_name,"_bin",bs,"_position.txt"),col.names=T,row.names=F,sep="\t",quote=F)
    # bkpos <- out
}

import rasterio
import numpy as np
from rasterio.transform import from_bounds
from Log import log

def generate_chm(dsm_path, dtm_path, output_path):
    try:
        with rasterio.open(dsm_path) as dsm:
            dsm_data = dsm.read(1)
            transform = dsm.transform
            crs = dsm.crs
            bounds = dsm.bounds
            profile = dsm.profile
        with rasterio.open(dtm_path) as dtm:
            dtm_data = dtm.read(1)
            if dtm.shape != dsm.shape:
                log("DSM and DTM shapes mismatch, resampling DTM", "warning")
                dtm_data = np.resize(dtm_data, dsm_data.shape)
        chm_data = dsm_data - dtm_data
        chm_data = np.where(chm_data < 0, 0, chm_data)  # Ensure non-negative heights
        chm_data = np.where(np.isfinite(chm_data), chm_data, 0)  # Replace NaN/inf
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(chm_data, 1)
        log(f"Generated CHM: {output_path}, max value: {chm_data.max()}", "success")
    except Exception as e:
        log(f"Failed to generate CHM: {str(e)}", "error")
        raise

if __name__ == "__main__":
    dsm_path = "C:/Users/gabri/Documents/ARBOR ANALYZER/TEST FILES/2025-07-02_Walker_Springs/DSM.tiff"
    dtm_path = "C:/Users/gabri/Documents/ARBOR ANALYZER/TEST FILES/2025-07-02_Walker_Springs/DTM.laz"
    output_path = "C:/Users/gabri/Documents/ARBOR ANALYZER/TEST FILES/2025-07-02_Walker_Springs/CHM_new.tiff"
    generate_chm(dsm_path, dtm_path, output_path)
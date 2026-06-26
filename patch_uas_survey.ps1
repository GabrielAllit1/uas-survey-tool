# Patch gcp_generator.py to use pycuda.autoinit and projected polygon for area
$gcpPath = "gcp_generator.py"
if (Test-Path $gcpPath) {
    $content = Get-Content $gcpPath
    # Replace manual context creation with autoinit import
    $content = $content -replace 'import pycuda.driver as cuda', 'import pycuda.autoinit'
    $content = $content -replace 'cuda\.Device\(\d+\)\.make_context\(\)', '# Removed manual context creation for autoinit'
    $content = $content -replace 'cuda_context\.pop\(\)', '# Removed manual context pop for autoinit'

    # Replace area calculation line with projected polygon coordinates
    $content = $content | ForEach-Object {
        if ($_ -match 'coords = \[\(x, y\) for x, y in polygon_wgs84\.exterior\.coords\]') {
            @"
utm_crs = get_utm_crs_for_polygon(polygon_wgs84)
transformer_to_utm = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
polygon_utm = shapely_transform(transformer_to_utm.transform, polygon_wgs84)
coords = [(x, y) for x, y in polygon_utm.exterior.coords]
logger.debug(f"Polygon projected area (m²): {calculate_area_acres(coords)[0]:.2f}")
"@
        } else {
            $_
        }
    }

    $content | Set-Content $gcpPath
    Write-Host "Patched gcp_generator.py"
} else {
    Write-Host "gcp_generator.py not found!"
}

# Patch modern_ui.py for projected polygon area calc logging
$modernPath = "modern_ui.py"
if (Test-Path $modernPath) {
    $content = Get-Content $modernPath

    $content = $content | ForEach-Object {
        if ($_ -match 'area_m2, area_acres, fam_scale, _ = calculate_area_acres\(coords\)') {
            @"
utm_crs = get_utm_crs_for_polygon(aoi_polygon)
transformer_to_utm = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
polygon_utm = shapely_transform(transformer_to_utm.transform, aoi_polygon)
coords = [(x, y) for x, y in polygon_utm.exterior.coords]
area_m2, area_acres, fam_scale, _ = calculate_area_acres(coords)
logger.debug(f"AOI projected area: {area_m2:.2f} m², {area_acres:.2f} acres")
"@
        } else {
            $_
        }
    }

    $content | Set-Content $modernPath
    Write-Host "Patched modern_ui.py"
} else {
    Write-Host "modern_ui.py not found!"
}

# Patch filter_coordinates.py example (add import)
$filterPath = "filter_coordinates.py"
if (Test-Path $filterPath) {
    $content = Get-Content $filterPath

    # Add import if missing
    if (-not ($content -join "`n" -match 'from pyproj import Transformer')) {
        $content = @("from pyproj import Transformer") + $content
    }

    $content | Set-Content $filterPath
    Write-Host "Patched filter_coordinates.py (added pyproj import)"
} else {
    Write-Host "filter_coordinates.py not found!"
}

Write-Host "Auto patch complete! Please review changes before running."

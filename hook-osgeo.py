from PyInstaller.utils.hooks import collect_data_files, collect_submodules
datas = collect_data_files('osgeo') + collect_data_files('osgeo.gdal') + collect_data_files('osgeo.ogr') + collect_data_files('osgeo.osr')
datas += collect_data_files('gdal', include_py_files=True)
datas += [(r"C:\ProgramData\miniconda3\envs\uas_survey_tool_v2\Library\share\gdal", "gdal"),
          (r"C:\ProgramData\miniconda3\envs\uas_survey_tool_v2\Library\share\proj", "proj"),
          (r"C:\ProgramData\miniconda3\envs\uas_survey_tool_v2\Library\bin", "Library/bin")]
hiddenimports = collect_submodules('rasterio') + ['osgeo', 'osgeo.gdal', 'osgeo.ogr', 'osgeo.osr']
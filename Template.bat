@echo off
setlocal enabledelayedexpansion

:: 1) Activate conda
call conda activate dic

:: 2) Enter project directory
cd /d C:\01project\Mesh-DIC

:: 3) Data root path
set DATA_ROOT=C:\01project\DIC_boundary_test_data-main\case\circle\scale
set MESH_DIR=C:\01project\DIC_boundary_test_data-main\case\circle\mesh
set MESH_DIR_UNIX=!MESH_DIR:\=/!

echo -----------------------------------------------
echo   Start scanning path: %DATA_ROOT%
echo -----------------------------------------------
echo.

for /d %%D in ("%DATA_ROOT%\*") do (
    echo Found subfolder: %%D

    :: Windows paths
    set INPUT_DIR=%%D\
    set OUTPUT_DIR=%%D\Q8DIC\

    :: Only convert to UNIX when writing into JSON
    set INPUT_DIR_UNIX=!INPUT_DIR:\=/!
    set OUTPUT_DIR_UNIX=!OUTPUT_DIR:\=/!

    echo Using paths for JSON:
    echo     !INPUT_DIR_UNIX!
    echo     !OUTPUT_DIR_UNIX!
    echo.

    :: Create output dir (Windows path!)
    if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"

    :: Write JSON (use CALL to ensure expansion)
    > config.json echo {
    call echo     "input_dir": "!INPUT_DIR_UNIX!",>> config.json
    call echo     "output_dir": "!OUTPUT_DIR_UNIX!",>> config.json
    call echo     "mesh_dir": "!MESH_DIR_UNIX!",>> config.json

    >> config.json echo     "mesh_size": 41.0,
    >> config.json echo     "simplify_roi_boundary_poly": 2.0,
    >> config.json echo     "bcoef_border": 3,
    >> config.json echo     "max_iterations": 100,
    >> config.json echo     "cutoff_diffnorm": 1e-4,
    >> config.json echo     "lambda_reg": 1e-6,
    >> config.json echo     "displacement_init": "int_pixels",
    >> config.json echo     "subset_r": 21,
    >> config.json echo     "search_radius": 10,

    >> config.json echo     "parallel": true,
    >> config.json echo     "max_workers": 1,
    >> config.json echo     "smooth_flag": false,
    >> config.json echo     "smooth_method": "gaussian",
    >> config.json echo     "smooth_sigma": 1.0,

    >> config.json echo     "strain_calculate_flag": true,
    >> config.json echo     "strain_method": "gaussian_window",
    >> config.json echo     "strain_window_half_size": 51,

    >> config.json echo     "show_plot": true,
    >> config.json echo     "save_mesh_plot": true
    >> config.json echo }

    echo config.json updated
    echo.

    :: Run Python solve
    python DIC_main_solver.py

    echo Completed folder: %%D
    echo -----------------------------------------------
    echo.
)

echo All done!
pause

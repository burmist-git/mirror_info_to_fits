python mirror_info_to_fits.py astri_M1_segments_pre-production.dat astri_M1_segments_pre-production.fits primary_mirror_parameters-1.0.0.json SST --shape_segmentation
python3 -c "from astropy.table import Table; t = Table.read('astri_M1_segments_pre-production.fits'); print(t)"

python mirror_info_to_fits.py mirror_CTA-N-LST1_v2019-03-31.dat mirror_CTA-N-LST1_v2019-03-31.fits dish_shape_length-LST_test.json LST
python3 -c "from astropy.table import Table; t = Table.read('mirror_CTA-N-LST1_v2019-03-31.fits'); print(t)"

python mirror_info_to_fits.py mirror_CTA-100_1.20-86-0.04.dat mirror_CTA-100_1.20-86-0.04.fits dish_shape_length-1.0.0.json MST
python3 -c "from astropy.table import Table; t = Table.read('mirror_CTA-100_1.20-86-0.04.fits'); print(t)"


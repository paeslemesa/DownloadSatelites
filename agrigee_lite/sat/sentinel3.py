"""
DOWNLOAD SENTINEL-3 OLCI DATA GROM GOOGLE EARTH ENGINE

This script downloads Sentinel-3 OLCI data from Google Earth Engine (GEE) and saves it to a local directory as TIFF files.
The script uses the Google Earth Engine Python API to access the data and the rasterio library to save the data as TIFF files.
The script is designed to be run from the command line and takes the following arguments:

1. start_date: The start date of the time range for the data to be downloaded (YYYY-MM-DD).
2. end_date: The end date of the time range for the data to be downloaded (YYYY-MM-DD).
3. region: The region of interest for the data to be downloaded (e.g., 'region1', 'region2').
4. output_dir: The directory where the downloaded data will be saved.
5. band: The band of the Sentinel-3 OLCI data to be downloaded (e.g., 'Oa01_radiance', 'Oa02_radiance').
6. cloud_coverage: The maximum cloud coverage percentage for the data to be downloaded (0-100).
7. max_results: The maximum number of results to be returned (default is 1000).


References:
https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S3_OLCI?hl=pt-br
"""
#%%-------------------------------------------------------------------------------
# LIBRARIES
#---------------------------------------------------------------------------------
from pathlib import Path
import ee
import geemap
import datetime
import rasterio
import numpy as np
from tqdm import tqdm
import geopandas as gpd
import ee.mapclient


#%%-------------------------------------------------------------------------------
# INITIALIZE EARTH ENGINE
#---------------------------------------------------------------------------------
ee.Authenticate()
ee.Initialize(project='ee-sabrinamateusufv')

#%%-------------------------------------------------------------------------------
# INPUTS
#---------------------------------------------------------------------------------

wkdir = Path(r"/mnt/d/Doutorado_UFV/Artigo_Mestrado")

start_date   = '2023-08-18'  # Start date of the time range
end_date     = '2023-08-20'  # End date of the time range
regiona_name = "AOI Mestrado"  # Region of interest

shp_path = Path(wkdir, "aoi.geojson") # Path to the shapefile

#%%-------------------------------------------------------------------------------
# FUNCTIONS
#---------------------------------------------------------------------------------
def get_ee_geometry(shp_path:Path) -> ee.Geometry:
    """
    Get the bounding box of the shapefile.
    Args:
        shp_path (Path): Path to the shapefile in any CRS.
    Returns:
        ee.Geometry: Bounding box of the shapefile.
    
    """
    gdf = gpd.read_file(shp_path) # Read the shapefile

    gdf = gdf.to_crs(epsg=4326) # Convert to WGS84
    gdf = gdf.reset_index(drop=True) # Reset the index

    bounds = gdf.bounds.values[0] # Get the bounds of the shapefile

    bbox = [[bounds[0], bounds[1]], # Get bounding box
            [bounds[0], bounds[3]],
            [bounds[2], bounds[3]],
            [bounds[2], bounds[1]],
            [bounds[0], bounds[1]]]

    # Cria um dicionário da área de interesse para nossa pesquisa
    aoi = {
        "type": "Polygon",
        "coordinates": [bbox]
    }
    
    return ee.Geometry(aoi)


def multiply_scale_factor_image(image: ee.Image) -> ee.Image:
    """
    Multiply Sentinel-3 OLCI bands by their respective scale factors.
    Args:
        image (ee.Image): Image with raw radiance bands.
    Returns:
        ee.Image: Image with scaled bands (original names).
    """
    dict_scale_factors = {
        'Oa01_radiance': 0.01394650,
        'Oa02_radiance': 0.01338730,
        'Oa03_radiance': 0.01214810,
        'Oa04_radiance': 0.01151980,
        'Oa05_radiance': 0.01009530,
        'Oa06_radiance': 0.01235380,
        'Oa07_radiance': 0.00879161,
        'Oa08_radiance': 0.00876539,
        'Oa09_radiance': 0.00951030,
        'Oa10_radiance': 0.00773378,
        'Oa11_radiance': 0.00675523,
        'Oa12_radiance': 0.00719960,
        'Oa13_radiance': 0.00749684,
        'Oa14_radiance': 0.00865120,
        'Oa15_radiance': 0.00526779,
        'Oa16_radiance': 0.00530267,
        'Oa17_radiance': 0.00493004,
        'Oa18_radiance': 0.00549962,
        'Oa19_radiance': 0.00502847,
        'Oa20_radiance': 0.00326378,
        'Oa21_radiance': 0.00324118,
    }

    scaled_bands = [
        image.select(band).multiply(factor).rename(band)
        for band, factor in dict_scale_factors.items()
    ]

    return ee.Image.cat(scaled_bands).copyProperties(image, image.propertyNames())

    

def get_S3_dataset(aoi: ee.Geometry, start_date:str, end_date:str, bands = None) -> ee.ImageCollection:
    """
    Get the Sentinel-3 OLCI dataset from Google Earth Engine.
    Args:
        aoi (ee.Geometry): Area of interest.
        start_date (str): Start date of the time range.
        end_date (str): End date of the time range.
        cloud (int): Maximum cloud coverage percentage.
        bands (list): List of bands to be downloaded. If None, all bands will be downloaded.
    Returns:
        ee.ImageCollection: Sentinel-3 OLCI dataset.
    """
    # Get the Sentinel-3 OLCI dataset from Google Earth Engine
    dataset = ee.ImageCollection("COPERNICUS/S3/OLCI")


    # Filter the dataset by date and region
    filtered_dataset = dataset.filterDate(start_date, end_date).filterBounds(aoi)

    # Apply the scale factor to the dataset
    scaled_dataset = filtered_dataset.map(multiply_scale_factor_image)

    # Select the bands to be downloaded
    if bands is not None:
        scaled_dataset = scaled_dataset.select(bands)

    return scaled_dataset


def download_S3_image(aoi: ee.Feature, ds: ee.ImageCollection, region_name: str) -> None:
    """
    Download a Sentinel-3 OLCI image from Google Earth Engine and save it as a GeoTIFF file.
    Args:
        image (ee.Image): Sentinel-3 OLCI image.
        output_dir (Path): Directory to save the downloaded image.
        region_name (str): Name of the region of interest.
    """
    # Get the date of the image
    for i in tqdm(range(ds.size().getInfo()), desc="Downloading images", unit="image"):
        image = ds.toList(ds.size()).get(i) # Get the image from the dataset
        image = ee.Image(image) # Convert to ee.Image
        image = image.clip(aoi) # Clip the image to the area of interest

        image_id = image.get('system:index').getInfo()
        

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=image_id,
            fileNamePrefix=image_id,
            folder= f'ee_{region_name}',
            region=aoi,
            scale=300,
            crs='EPSG:4326'
        )
        task.start()


# %%---------------------------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------------------------

aoi = get_ee_geometry(shp_path) # Get the bounding box of the shapefile
ds = get_S3_dataset(aoi, start_date, end_date) # Get the Sentinel-3 OLCI dataset
download_S3_image(aoi, ds, region_name=regiona_name) # Download the images


# %%

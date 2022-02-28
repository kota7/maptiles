# -*- coding: utf-8 -*-

import os
import math
import sys
import sqlite3
import warnings
from io import BytesIO
from collections import namedtuple
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.scale import FuncScale
import pyproj

class config:
    dbfile = os.path.expanduser("~/maptiles.db")

def set_databasefile(filepath: str):
    # change the location of sqlite3 database
    config.dbfile = filepath

# ***   DATABASE   ********************************************************************** #
def initialize_database(replace=False):
    if replace and os.path.isfile(config.dbfile):
        os.unlink(config.dbfile)
    with sqlite3.connect(config.dbfile) as conn:
        c = conn.cursor()
        q = """CREATE TABLE IF NOT EXISTS tiles
               (url TEXT UNIQUE PRIMARY KEY, image BLOB)"""
        c.execute(q)
        conn.commit()

def _get_tileimage(url: str, use_cache: bool=True)-> Image:
    if not os.path.isfile(config.dbfile):
        initialize_database(replace=False)

    if use_cache:
        with sqlite3.connect(config.dbfile) as conn:
            c = conn.cursor()
            q = """SELECT image FROM tiles WHERE url = ?"""
            res = c.execute(q, (url,)).fetchall()
            if len(res) >= 1:
                # due to the unique constraint, this should match at most one row
                # but use `>=` instead of `==` just in case
                b = res[0][0] 
                img = Image.frombytes("RGB", (256, 256), b)
                # map tile math depends on all tiles are 256x256.
                return img
    # could not find the image in the database, so download from the server
    r = requests.get(url)
    if r.status_code in (200, 201, 202):  # choice of success codes, can be arbitrary
        #print(r.status_code, url)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        # save to the database as bytes
        with sqlite3.connect(config.dbfile) as conn:
            c = conn.cursor()
            q = """INSERT OR REPLACE INTO tiles VALUES (?,?)"""
            c.execute(q, (url, img.tobytes()))
            conn.commit()    
    else:
        warnings.warn("Failed to fetch tile image from {} with status code {}".format(
            url, r.status_code))
        # download failed, use white image
        img = Image.new("RGB", (256, 256), (255, 255, 255))
    
    return img
# ***   END OF DATABASE   ****************************************************************** #


# ***   TILES   **************************************************************************** #
_Tile = namedtuple("Tile", "name baseurl copyright copyright_html")
def Tile(baseurl, name=None, copyright=None, copyright_html=None):
    return _Tile(name=name, baseurl=baseurl, copyright=copyright, copyright_html=copyright_html)

class _tiles:

    # OpenStreetMap
    # https://wiki.openstreetmap.org/wiki/Tile_servers
    _osm_copyright = ("\u00a9 OpenStreetMap contributors",
                      '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors')

    osm = Tile(
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "OpenStreetMap, Standard",
        *_osm_copyright)
    osm_bw = Tile(
        "http://tiles.wmflabs.org/bw-mapnik/{z}/{x}/{y}.png", 
        "OpenStreetMap, Black&White",
        *_osm_copyright)
    osm_tonner = Tile(
        "http://tile.stamen.com/toner/{z}/{x}/{y}.png", 
        "OpenStreetMap, Toner",
        *_osm_copyright)
    osm_tonner_hybrid = Tile(
        "http://tile.stamen.com/toner-hybrid/{z}/{x}/{y}.png", 
        "OpenStreetMap, Toner Hybrid",
        *_osm_copyright)
    osm_tonner_labels = Tile(
        "http://tile.stamen.com/toner-labels/{z}/{x}/{y}.png", 
        "OpenStreetMap, Toner Labels",
        *_osm_copyright)
    osm_tonner_lines = Tile(
        "http://tile.stamen.com/toner-lines/{z}/{x}/{y}.png", 
        "OpenStreetMap, Toner Lines",
        *_osm_copyright)
    osm_tonner_background = Tile(
        "http://tile.stamen.com/toner-background/{z}/{x}/{y}.png", 
        "OpenStreetMap, Toner Backgrounds",
        *_osm_copyright)
    osm_tonner_lite = Tile(
        "http://tile.stamen.com/toner-lite/{z}/{x}/{y}.png", 
        "OpenStreetMap, Toner Lite",
        *_osm_copyright)

    # Geospatial Information Authority of Japan
    # https://maps.gsi.go.jp/development/ichiran.html
    _japangsi_copyright = (
        "\u00a9 国土地理院 | Geospatial Information Authority of Japan",
        '&copy; <a href="https://maps.gsi.go.jp/development/ichiran.html">国土地理院 | Geospatial Information Authority of Japan</a>')

    japangsi = Tile(
        "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
        "Geospatial Information Authority of Japan, Standard",
        *_japangsi_copyright)
    japangsi_pale = Tile(
        "https://cyberjapandata.gsi.go.jp/xyz/pale/{z}/{x}/{y}.png",
        "Geospatial Information Authority of Japan, Pale",
        *_japangsi_copyright)
    japangsi_blank = Tile(
        "https://cyberjapandata.gsi.go.jp/xyz/blank/{z}/{x}/{y}.png",
        "Geospatial Information Authority of Japan, Blank",
        *_japangsi_copyright)

    # Google Maps
    # ToDo: check copyright information
    _google_copyright = ("\u00a9 Google", '&copy; <a href="https://google.com">Google</a>')
    google = Tile(
        "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        "Google Map",
        *_google_copyright)
    google_roads = Tile(
        "https://mt1.google.com/vt/lyrs=h&x={x}&y={y}&z={z}",
        "Google Map, Road",
        *_google_copyright)
    google_streets = Tile(
        "https://mt1.google.com/vt/lyrs=r&x={x}&y={y}&z={z}",
        "Google Map, Streets",
        *_google_copyright)
    google_terrain = Tile(
        "https://mt1.google.com/vt/lyrs=t&x={x}&y={y}&z={z}",
        "Google Map, Terrain",
        *_google_copyright)
    google_satellite = Tile(
        "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "Google Map, Satellite",
        *_google_copyright)
    google_satellite_hybrid = Tile(
        "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "Google Map, Satellite Hybrid",
        *_google_copyright)
    # alias
    google_h = google_roads
    google_r = google_streets
    google_t = google_terrain
    google_s = google_satellite
    google_y = google_satellite_hybrid

def predefined_tiles():
    out = vars(_tiles)
    return {key:value for key, value in out.items() if type(value)==_Tile}

def get_tile(name: str)-> _Tile:    
    return predefined_tiles().get(name, None)
# ***   END OF TILES   ********************************************************************** #


# ***   MATH   ****************************************************************************** #
L = 180.0/math.pi*math.asin(math.tanh(math.pi))  # latitude limit both north and south end

# Convert WGS84 lon-lat to pixel index
def _lon_to_pixel(lon: float, z: int)-> float:
    # make sure lon is in -180, 180
    lon %= 360
    if lon >= 180: lon -= 360
    return 2**(z+7) * (lon / 180.0 + 1.0)

def _lat_to_pixel(lat: float, z: int)-> float:
    assert lat >= -L and lat <= L
    #return 2**(z+7) / math.pi * (-math.atanh(math.sin(lat/180.0*math.pi)) + math.atanh(math.sin(math.pi/180.0*L)))
    return 2**(z+7) / math.pi * (math.pi-math.atanh(math.sin(lat/180.0*math.pi)))

# numpy version of latitude conversion
# used for axis scaling
def _lats_to_y(lats: np.array)-> np.array:
    # latitudes in degree to web map y id, unscaled
    #return -np.degrees(np.pi - np.log(np.tan(np.pi/4.0 + np.radians(lats)/2.0) + 1e-6)) / 2.0 * L / 90.0
    lats = np.asarray(lats)     # robust just in case a list is given
    lats[np.isinf(lats)] = 0    # both inf, -inf to 0, to avoid runtime warnings
    #print(np.unique(rad))
    return np.pi - np.arctanh(np.sin(np.radians(lats)))

def _y_to_lats(y: np.array)-> np.array:
    # inverse of _lats_to_y
    return np.degrees(np.arcsin(np.tanh(np.pi-y)))

# Convert WGS84 lon-lat to tile and within-tile index
def _get_tile_index(lon: float, lat: float, z: int)-> tuple:
    p, q = _lon_to_pixel(lon, z), _lat_to_pixel(lat, z)
    # tile id
    x, y = int(p/256), int(q/256)
    # within tile index
    i, j = int(p % 256), int(q % 256)
    return (x, y, i, j)

def _pixel_to_lon(p: float, z: int)-> float:
    return 180.0 * (p/2.0**(z+7) - 1)

def _pixel_to_lat(q: float, z: int)-> float:
    #return 180.0/math.pi * (math.asin(math.tanh(-math.pi/2**(z+7)*q + math.atanh(math.sin(math.pi/180.0*L)))))
    return 180.0/math.pi * (math.asin(math.tanh(-math.pi/2**(z+7)*q + math.pi)))

def _get_extent(x1, x2, y1, y2, i1, i2, j1, j2, z, crs="epsg:4326"):
    lon1 = _pixel_to_lon(x1*256+i1, z)
    lon2 = _pixel_to_lon(x2*256+i2+1, z)
    lat1 = _pixel_to_lat(y1*256+j1, z)
    lat2 = _pixel_to_lat(y2*256+j2+1, z)

    crs = crs.lower()
    if crs != "epsg:4326":
        t = pyproj.Transformer.from_crs("epsg:4326", crs, always_xy=True)
        lon1, lat1 = t.transform(lon1, lat1)
        lon2, lat2 = t.transform(lon2, lat2)

    #return lon1, lon2, lat1, lat2
    return lon1, lon2, lat2, lat1

# def _great_circle_distance(lon1, lat1, lon2, lat2, radius=6371):
#   # https://www.movable-type.co.uk/scripts/latlong.html
#   lat1, lat2, lon1, lon2 = math.radians(lat1), math.radians(lat2), math.radians(lon1), math.radians(lon2)
#   dlat, dlon = lat2 - lat1, lon2 - lon1
#   a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#   c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
#   return radius * c

# def _estimate_aspect(lon1, lat1, lon2, lat2)-> float:
#     # Calculate the ratio of disance per degree (dist-per-lat/dist-per-lon)
#     lat_m = (lat1+lat2)/2
#     lon_m = (lon1+lon2)/2
#     # horizontal distance at the middle latitude
#     dist_lon = _great_circle_distance(lon1, lat_m, lon2, lat_m)
#     print(dist_lon)
#     # vertical distance at the middle longitude
#     dist_lat = _great_circle_distance(lon_m, lat1, lon_m, lat2)
#     print(dist_lat)
#     return abs(dist_lat / (lat2-lat1) * (lon2-lon1) / dist_lon)
class WebMercatorLatitudeScale:
    """
    Axis scaler to project latitudes to the web mercator scale
    For plotting compatibility
    """
    def __init__(self, top: float, bottom: float):
        self.top = top
        self.bottom = bottom
        ylim = _lats_to_y([top, bottom])
        #print(ylim)
        self.slope = (top-bottom)/(ylim[0]-ylim[1])
        self.const = self.top - self.slope * ylim[0]
        #print(self.slope, self.const)
    
    def forward(self, lats):
        return self.const + self.slope* _lats_to_y(lats)
    
    def inverse(self, y):
        return _y_to_lats((y-self.const)/self.slope)

def web_mercator_yscale(top: float, bottom: float):
    s = WebMercatorLatitudeScale(top, bottom)
    return (s.forward, s.inverse)
# alias. this could be more intuitive because web mercator is de facto standard for web mapping
web_map_yscale = web_mercator_yscale
# ***   END OF MATH   ************************************************************************** #


def _auto_zoom(lon1, lon2)-> int:
    # Heuristic determination of zoom level
    #   we find z such that: 360/2^z ~ width/2
    #   i.e. roughly two tiles to cover the width
    lon1 = lon1 % 360
    lon2 = lon2 % 360
    width = lon2 - lon1
    if width < 0:
        # case when the range overlaps the longitude zero point.
        # e.g. 350 to 20 --> -330 + 360 = 30
        width += 360 
    return round(math.log2(360*2.0) - math.log2(width))

def _crs_from_str(s):
    s = s.lower()
    if s in ("lonlat", "wgs84", "epsg:4326"):
        return "epsg:4326"
    elif s in ("webmap", "web-mercator", "epsg:3857"):
        return "epsg:3857"
    else:
        raise ValueError("'%' is not supported CRS string" % s)

def get_maparray(bounds: tuple, tile: _Tile="osm", z: int=None, extent_crs="lonlat", use_cache:bool =True, verbose: int=1)-> tuple:
    """
    Draw map on matlotlib axes.

    Args:
        bounds     : Tuple of floats (lon1, lat1, lon2, lat2), that defines the rectangle area to draw.
        tile       : Str or _Tile object. 
                     If str, must be either
                      - One of the predefined tile ids. Use predefined_tiles() to see the available tiles.
                      - Map url template with {x} {y} {z} parameters.
                      See Tile() to generate a _Tile object.
        z          : Zoome level. If None, heuristically chosen.
        extent_crs : Str (case insensitive) indicating the coordinate reference system of the extent returned.
                     - "lonlat", "wgs84" or "epsg:4326"        --> standard longitude, latitude in WGS84 system.
                     - "webmap", "web-mercator" or "epsg:3857" --> indices in EPSG:3857 system (Web Mercator)
        use_cache  : Bool. If True, map images stored in the internal database are used. This helps to reduce the
                    number of web requests to the map tile servers.
        verbose  : Int. 0: print no message, 1: print the auto-chozen zoom level, 2 or larger: show the url of the tiles
    
    Returns:
        Tuple of
        - Numpy array of shape (height, width, 3). RGB image data.
        - Tuple of (xmin, xmax, ymin, ymax) defining the covered area of the image
    """
    if type(tile)==str:
        if hasattr(_tiles, tile):
            # predefined tile name
            tile = getattr(_tiles, tile)
        elif all(("{%s}" % a) in tile for a in "xyz"):
            # contains {x} {y} {z}, so use this as baseurl
            tile = Tile(tile)
        else:
            raise ValueError("Invalid tile string: '{}'".format(tile))

    lon1, lat1, lon2, lat2 = bounds
    # lon1 > lon2 is unlikely but possible for cases overlapping the meridian
    #if lon1 > lon2: lon1, lon2 = lon2, lon1  
    if lat1 < lat2:
        lat1, lat2 = lat2, lat1
    # make sure (lon1, lat1) is top-left, (lon2, lat2) is bottom-right
    if z is None:
        z = _auto_zoom(lon1, lon2)
        if verbose >= 1:
            print("Zoom level %d is chosen" % z, file=sys.stderr)

    # find tiles of the corners
    (x1, y1, i1, j1) = _get_tile_index(lon1, lat1, z)
    (x2, y2, i2, j2) = _get_tile_index(lon2, lat2, z)
    #print(x1, x2, y1, y2)
    #print(i1, i2, j1, j2)
    out = []
    # for edge case where the range overlaps the meridian
    xs = range(x1, x2+1) if x1 <= x2 else list(range(x1, 2**z)) + list(range(0, x2+1))
    for y in range(y1, y2+1):
        row = []
        for x in xs:
            url = tile.baseurl.format(x=x, y=y, z=z)
            if verbose >= 2:
                print(url, file=sys.stderr)
            img = _get_tileimage(url, use_cache=use_cache)
            row.append(np.asarray(img))
        out.append(np.concatenate(row, axis=1))
    out = np.concatenate(out, axis=0)
    # truncate to the bounds positions
    j2b = 256*(y2-y1) + j2 + 1
    i2b = 256*(x2-x1) + i2 + 1 
    out = out[j1:j2b, i1:i2b]
    # calculate extent of the resulted image
    crs = _crs_from_str(extent_crs)
    extent = _get_extent(x1, x2, y1, y2, i1, i2, j1, j2, z, crs=crs)
    return out, extent

def draw_map(bounds: tuple, tile: Tile="osm", z: int=None, aspect="auto",
             extent_crs="lonlat", scaling: bool=False,
             use_cache:bool =True, verbose: int=1, ax=None, **kwargs)-> tuple:
    """
    Draw map on matlotlib axes.

    Args:
        bounds   : Tuple of floats (lon1, lat1, lon2, lat2), that defines the rectangle area to draw.
        tile     : Str or _Tile object. 
                    If str, must be either
                    - One of the predefined tile ids. Use predefined_tiles() to see the available tiles.
                    - Map url template with {x} {y} {z} parameters.
                    See Tile() to generate a _Tile object.
        z        : Zoome level. If None, heuristically chosen.
        aspect   : Aspect ratio (lat / lon). If "auto", heuristically chosen.
        scaling  : Bool that indicates if the y-axis should be scaled by web mercator projection.
                   If True, the map image is drawn on a twinx axes (with no scaling), 
                   and y-axis of the main axes will be scaled by web mercator with top and bottom latitude shared.
                   If False, the map image is drawn on the main axes. 
        extent_crs : Str (case insensitive) indicating the coordinate reference system of the image extent.
                     - "lonlat", "wgs84" or "epsg:4326"        --> standard longitude, latitude in WGS84 system.
                     - "webmap", "web-mercator" or "epsg:3857" --> indices in EPSG:3857 system (Web Mercator)
        use_cache: Bool. If True, map images stored in the internal database are used. This helps to reduce the
                    number of web requests to the map tile servers.
        verbose  : Int. 0: print no message, 1: print the auto-chozen zoom level, 2 or larger: show the url of the tiles
        ax       : If given, map image is to drawn on this axes, and autoscale is disabled.
        kwargs   : Optional arguments passed to ax.imshow() or plt.imshow()
    
    Returns:
        Tuple of two matplotlib Axes.
        - Axes object
        - AxesImage object
    """
    array, extent = get_maparray(bounds, tile, z, use_cache=use_cache, extent_crs=extent_crs, verbose=verbose)
    #print(extent)
    if "extent" in kwargs:
        extent = kwargs.pop("extent")
    if aspect=="auto":
        #aspect = _estimate_aspect(*bounds)
        # adjust aspect so that the each pixel is a square of the same size
        aspect = 1.0 * array.shape[0] / array.shape[1] * abs(extent[0]-extent[1]) / abs(extent[2]-extent[3]) 

    #print(extent)
    if ax is None:        
        ax = plt.gca()
        #print(opts)
    
    opts = {"extent": extent, "aspect": aspect}
    opts.update(kwargs)  # if extent and/or aspect is supplied, replace
    if scaling:
        # add another axes only for drawing the image
        ax2 = ax.twinx()
        ax_img = ax2.imshow(array, **opts)
        ax2.yaxis.set_visible(False)

        # move the main axes to the top layer and its patch invisible
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

        # Apply web-mercator scale to the the main axes
        # with the same ylimit and aspect
        yscale = web_mercator_yscale(extent[2], extent[3])
        ax.set_yscale("function", functions=yscale)
        ax.set_ylim((extent[2], extent[3]), auto=False)
        ax.set_aspect(aspect, share=True)
    else:
        ax_img = ax.imshow(array, **opts)
        ax.set_ylim((extent[2], extent[3]), auto=False)
    
    return ax, ax_img

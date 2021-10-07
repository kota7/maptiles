# -*- coding: utf-8 -*-

__version__ = "0.0.1"

import os
import math
import sys
import sqlite3
from io import BytesIO
from collections import namedtuple
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt

class config:
    dbfile = os.path.expanduser("~/maptiles.db")

def set_databsefile(filepath):
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

def get_tileimage(url)-> Image:
    if not os.path.isfile(config.dbfile):
        initialize_database(replace=False)

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
    if r.status_code in (200, 201, 202):  # choice of success codes
        img = Image.open(BytesIO(r.content)).convert("RGB")
    else:
        # download failed, use white image
        img = Image.new("RGB", (256, 256), (255, 255, 255))
    
    # save to the database as bytes
    with sqlite3.connect(config.dbfile) as conn:
        c = conn.cursor()
        q = """INSERT OR REPLACE INTO tiles VALUES (?,?)"""
        c.execute(q, (url, img.tobytes()))
        conn.commit()    
    return img
# ***   END OF DATABASE   ****************************************************************** #


# ***   TILES   **************************************************************************** #
Tile = namedtuple("Tile", "baseurl copyright copywright_html")

_tiles = {
   "osm": Tile("https://tile.openstreetmap.org/{z}/{x}/{y}.png", 
               "(c) OpenStreetMap contributors",
               '(c) <a href="http://openstreetmap.org">OpenStreetMap</a> contributors')
  , # to be added more
}
# ***   END OF TILES   ********************************************************************** #


# ***   MATH   ****************************************************************************** #
# Convert WGS84 lon-lat to pixel index
def _lon_to_pixel(lon: float, z: int)-> float:
    # make sure lon is in -180, 180
    lon %= 360
    if lon > 180: lon -= 360
    return 2**(z+7) * (lon / 180.0 + 1.0)

L = 180.0/math.pi*math.asin(math.tanh(math.pi))
def _lat_to_pixel(lat: float, z: int)-> float:
    assert lat >= -L and lat <= L
    return 2**(z+7) / math.pi * (-math.atanh(math.sin(lat/180.0*math.pi)) + math.atanh(math.sin(math.pi/180.0*L)))

# Convert WGS84 lon-lat to tile and within-tile index
def _get_tile_index(lon: float, lat: float, z: int)-> tuple:
    p, q = _lon_to_pixel(lon, z), _lat_to_pixel(lat, z)
    # tile id
    x, y = int(p/256), int(q/256)
    # within tile index
    i, j = int(p % 256), int(q % 256)
    return (x, y, i, j)

def _pixel_to_lon(p: float, z: int)-> float:
    return 180.0 * (p/2**(z+7) - 1)

def _pixel_to_lat(q: float, z: int)-> float:
    return 180.0/math.pi * (math.asin(math.tanh(-math.pi/2**(z+7)*q + math.atanh(math.sin(math.pi/180.0*L)))))

def _get_extent(x1, x2, y1, y2, i1, i2, j1, j2, z):
    lon1 = _pixel_to_lon(x1*256+i1, z)
    lon2 = _pixel_to_lon(x2*256+i2+1, z)
    lat1 = _pixel_to_lat(y1*256+j1, z)
    lat2 = _pixel_to_lat(y2*256+j2+1, z)
    return lon1, lon2, lat1, lat2
# ***   END OF MATH   ************************************************************************** #


def _auto_zoom(lon1, lon2)-> int:
    # Heuristic determination of zoom level
    #   we find z such that: 360/2^z ~ width/2
    #   i.e. roughly two tiles to cover the width
    width = abs(lon2 - lon1)
    return round(math.log2(360*2.0) - math.log2(width))
    
def get_maparray(bounds: tuple, tile: Tile="osm", z: int=None):
    if type(tile)==str:
        assert tile in _tiles, "Tile '{}' is not defined".format(tile)
        tile = _tiles[tile]

    lon1, lon2, lat1, lat2 = bounds
    if lon1 > lon2: lon1, lon2 = lon2, lon1
    if lat1 < lat2: lat1, lat2 = lat2, lat1
    # make sure (lon1, lat1) is top-left, (lon2, lat2) is bottom-right
    if z is None:
        z = _auto_zoom(lon1, lon2)
        print("Zoom level %d is chosen" % z, file=sys.stderr)

    # find tiles of the corners
    (x1, y1, i1, j1) = _get_tile_index(lon1, lat1, z)
    (x2, y2, i2, j2) = _get_tile_index(lon2, lat2, z)
    #print(x1, x2, y1, y2)
    #print(i1, i2, j1, j2)
    out = []
    for y in range(y1, y2+1):
        row = []
        for x in range(x1, x2+1):
            url = tile.baseurl.format(x=x, y=y, z=z)
            img = get_tileimage(url)
            row.append(np.asarray(img))
        out.append(np.concatenate(row, axis=1))
    out = np.concatenate(out, axis=0)
    # truncate to the bounds positions
    out = out[j1:-(255-j2),i1:-(255-i2)]
    # calculate extent of the resulted image
    extent = _get_extent(x1, x2, y1, y2, i1, i2, j1, j2, z)
    return out, extent

def draw_map(bounds: tuple, tile: Tile="osm", z: int=None, ax=None, **kwargs):
    array, extent = get_maparray(bounds, tile, z)
    opts = {"extent": extent}
    opts.update(kwargs)  # if extent is supplied, use it
    if ax is None:
        plt.imshow(array, **opts)
    else:
        ax.imshow(array, **opts)
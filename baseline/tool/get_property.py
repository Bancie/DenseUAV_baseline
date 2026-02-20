"""GPS metadata extraction and reverse-geocoding utilities.

Provides helpers for:
- Converting GPS coordinates from DMS (degrees/minutes/seconds) format to
  decimal degrees.
- Extracting GPS and date EXIF tags from image files.
- Reverse-geocoding GPS coordinates to a structured street address using the
  Baidu Maps Geocoding API.

This module is intended to be imported by other scripts; it has no CLI
entry point::

    from tool.get_property import find_GPS_image, find_address_from_GPS
"""

import exifread
import re
import json
import requests

def latitude_and_longitude_convert_to_decimal_system(*arg):
    """Convert a DMS (degrees, minutes, seconds) coordinate to decimal degrees.

    Accepts the three components as separate positional arguments. The seconds
    value may be supplied as a fraction string (e.g. ``'3600/100'``).

    Args:
        *arg: Exactly three values — ``degrees``, ``minutes``, ``seconds``.
            ``seconds`` may be a string in the form ``'numerator/denominator'``.

    Returns:
        float: The coordinate expressed as decimal degrees.

    Example:
        >>> latitude_and_longitude_convert_to_decimal_system(39, 54, '3600/100')
        40.0
    """
    return float(arg[0]) + ((float(arg[1]) + (float(arg[2].split('/')[0]) / float(arg[2].split('/')[-1]) / 60)) / 60)

def find_GPS_image(pic_path):
    """Extract GPS and date EXIF metadata from an image file.

    Reads EXIF tags using ``exifread`` and parses latitude, longitude, altitude,
    their reference directions, and capture date.  Coordinates are returned in
    decimal degrees when possible, otherwise as raw integer tuples.

    Args:
        pic_path (str): Path to the image file (JPEG, TIFF, etc.) from which
            EXIF data will be read.

    Returns:
        dict: A dictionary with two keys:

            - ``'GPS_information'`` (dict): May contain any subset of the
              following keys: ``'GPSLatitudeRef'``, ``'GPSLongitudeRef'``,
              ``'GPSAltitudeRef'``, ``'GPSLatitude'``, ``'GPSLongitude'``,
              ``'GPSAltitude'``.  Latitude/longitude values are either a
              3-tuple of integers ``(deg, min, sec)`` or a single decimal
              ``float``.
            - ``'date_information'`` (str): Capture date string, or ``''``
              if no date tag is found.
    """
    GPS = {}
    date = ''
    with open(pic_path, 'rb') as f:
        tags = exifread.process_file(f)
        for tag, value in tags.items():
            if re.match('GPS GPSLatitudeRef', tag):
                GPS['GPSLatitudeRef'] = str(value)
            elif re.match('GPS GPSLongitudeRef', tag):
                GPS['GPSLongitudeRef'] = str(value)
            elif re.match('GPS GPSAltitudeRef', tag):
                GPS['GPSAltitudeRef'] = str(value)
            elif re.match('GPS GPSLatitude', tag):
                try:
                    match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                    GPS['GPSLatitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                except:
                    deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                    GPS['GPSLatitude'] = latitude_and_longitude_convert_to_decimal_system(deg, min, sec)
            elif re.match('GPS GPSLongitude', tag):
                try:
                    match_result = re.match('\[(\w*),(\w*),(\w.*)/(\w.*)\]', str(value)).groups()
                    GPS['GPSLongitude'] = int(match_result[0]), int(match_result[1]), int(match_result[2])
                except:
                    deg, min, sec = [x.replace(' ', '') for x in str(value)[1:-1].split(',')]
                    GPS['GPSLongitude'] = latitude_and_longitude_convert_to_decimal_system(deg, min, sec)
            elif re.match('GPS GPSAltitude', tag):
                GPS['GPSAltitude'] = str(value)
            elif re.match('.*Date.*', tag):
                date = str(value)
    return {'GPS_information': GPS, 'date_information': date}

def find_address_from_GPS(GPS):
    """Reverse-geocode GPS coordinates to a structured address via the Baidu Maps API.

    Calls the Baidu Maps Geocoding v2 REST endpoint with the latitude and
    longitude extracted from ``GPS['GPS_information']`` and returns the parsed
    address components.

    Args:
        GPS (dict): Dictionary as returned by :func:`find_GPS_image`.  The
            key ``'GPS_information'`` must contain ``'GPSLatitude'`` and
            ``'GPSLongitude'`` entries.

    Returns:
        str | tuple: If ``GPS['GPS_information']`` is empty, returns the
        string ``'This photo has no GPS information'``.  Otherwise returns a
        4-tuple ``(formatted_address, province, city, district)`` where each
        element is a string from the Baidu Maps API response.
    """
    secret_key = 'zbLsuDDL4CS2U0M4KezOZZbGUY9iWtVf'
    if not GPS['GPS_information']:
        return 'This photo has no GPS information'
    lat, lng = GPS['GPS_information']['GPSLatitude'], GPS['GPS_information']['GPSLongitude']
    baidu_map_api = "http://api.map.baidu.com/geocoder/v2/?ak={0}&callback=renderReverse&location={1},{2}s&output=json&pois=0".format(
        secret_key, lat, lng)
    response = requests.get(baidu_map_api)
    content = response.text.replace("renderReverse&&renderReverse(", "")[:-1]
    baidu_map_address = json.loads(content)
    formatted_address = baidu_map_address["result"]["formatted_address"]
    province = baidu_map_address["result"]["addressComponent"]["province"]
    city = baidu_map_address["result"]["addressComponent"]["city"]
    district = baidu_map_address["result"]["addressComponent"]["district"]
    return formatted_address,province,city,district

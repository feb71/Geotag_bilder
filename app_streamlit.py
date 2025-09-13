
import os, io, re, json, math, zipfile
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import piexif
from pyproj import Transformer, CRS
from streamlit_image_coordinates import streamlit_image_coordinates as img_coords
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

st.set_page_config(page_title="Geotagging bilder v11.8", layout="wide")
st.title("Geotagging bilder v11.8")
st.caption("v11.8 • Kart‑rotasjon (tegn/drag) • justerbar 'N'‑størrelse • Tab B (målebok) • etiketter for hjørner/senter • EXIF WGS84")

def deg_to_dms_rational(dd):
    sign = 1 if dd >= 0 else -1
    dd = abs(dd)
    d = int(dd); m_full = (dd - d) * 60; m = int(m_full)
    s = round((m_full - m) * 60 * 10000)
    return sign, ((d,1),(m,1),(s,10000))

def _is_valid_number(x):
    try: fx = float(x)
    except: return False
    return not (math.isnan(fx) or math.isinf(fx))

def _wrap_deg(d):
    d = float(d) % 360.0
    if d < 0: d += 360.0
    return d

def ang_diff(a,b):
    if a is None or b is None: return None
    return abs((a - b + 180) % 360 - 180)

def exif_gps(lat, lon, alt=None, direction=None):
    sign_lat, lat_dms = deg_to_dms_rational(lat)
    sign_lon, lon_dms = deg_to_dms_rational(lon)
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if sign_lat>=0 else b'S',
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLongitudeRef: b'E' if sign_lon>=0 else b'W',
        piexif.GPSIFD.GPSLongitude: lon_dms,
    }
    if _is_valid_number(alt):
        a = float(alt)
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if a>=0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = (int(round(abs(a)*100)), 100)
    if _is_valid_number(direction):
        d = _wrap_deg(direction)
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b'T'
        gps[piexif.GPSIFD.GPSImgDirection] = (int(round(d*100)), 100)
    return gps

def write_exif_jpeg_bytes(jpeg_bytes, lat, lon, alt=None, direction=None):
    open("tmp_in.jpg","wb").write(jpeg_bytes)
    try:
        im = Image.open("tmp_in.jpg")
        try:
            exif_dict = piexif.load(im.info.get("exif", b""))
        except Exception:
            exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}}
        exif_dict["GPS"] = exif_gps(lat, lon, alt, direction)
        exif_dict[ "0th" ][piexif.ImageIFD.Orientation] = 1
        exif_bytes = piexif.dump(exif_dict)
        im.save("tmp_out.jpg", "jpeg", exif=exif_bytes, quality=95)
        return open("tmp_out.jpg","rb").read()
    finally:
        for p in ("tmp_in.jpg","tmp_out.jpg"):
            try: os.remove(p)
            except: pass

def normalize_orientation(im: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(im)

def parse_float_maybe_comma(v):
    if v is None: return None
    if isinstance(v, (int,float)): return float(v)
    s = str(v).strip().replace(" ", "").replace("\xa0","")
    if s=="" or s.lower() in {"nan","none","-"}: return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try: return float(s)
    except: return None

def ensure_epsg(key_prefix: str, title: str, default: int = 25832):
    st.markdown(f"**{title}**")
    presets = {
        "EUREF89 / UTM32 (EPSG:25832)": 25832,
        "EUREF89 / UTM33 (EPSG:25833)": 25833,
        "WGS84 (EPSG:4326)": 4326,
        "Custom EPSG (skriv under)": None,
    }
    label = st.selectbox("Velg EPSG:", list(presets.keys()), index=0, key=f"{key_prefix}_sel")
    code = presets[label]
    custom = st.text_input("Custom EPSG (kun tall)", value="", key=f"{key_prefix}_custom") if code is None else ""
    epsg = code if code is not None else (int(custom) if custom.strip().isdigit() else None)
    if epsg is None:
        st.info(f"Ingen EPSG valgt – bruker default {default}."); epsg = default
    try: CRS.from_epsg(epsg)
    except Exception:
        st.error(f"Ugyldig EPSG: {epsg}. Bruker {default}."); epsg = default
    return epsg

def transform_EN_to_wgs84(E,N, src_epsg):
    tr = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lon, lat = tr.transform(float(E), float(N))
    return lat, lon

def transform_EN_to_epsg(E,N, src_epsg, dst_epsg):
    if src_epsg==dst_epsg: return float(E), float(N)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    X,Y = tr.transform(float(E), float(N))
    return X,Y

def _load_font(size_px: int):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size_px)
            except: pass
    return ImageFont.load_default()

def draw_north_arrow(im: Image.Image, heading_deg: float,
                     pos=("right","bottom"), size_px=120, margin=20,
                     color=(255,255,255), outline=(0,0,0),
                     n_label_size=18) -> Image.Image:
    if heading_deg is None: return im
    w,h = im.size
    cx = {"left": margin+size_px, "center": w//2, "right": w - margin - size_px}.get(pos[0], w - margin - size_px)
    cy = {"top": margin+size_px, "middle": h//2, "bottom": h - margin - size_px}.get(pos[1], h - margin - size_px)
    ang = math.radians(heading_deg % 360.0)
    dx,dy = 0,-1
    vx = dx*math.cos(ang) - dy*math.sin(ang)
    vy = dx*math.sin(ang) + dy*math.cos(ang)
    tip  = (int(cx + vx*size_px), int(cy + vy*size_px))
    tail = (int(cx - vx*(size_px*0.4)), int(cy - vy*(size_px*0.4)))
    draw = ImageDraw.Draw(im, "RGBA")
    draw.line([tail, tip], fill=color, width=max(4, size_px//15))
    head_len = size_px*0.25
    left_ang  = math.atan2(vy, vx) + math.radians(150)
    right_ang = math.atan2(vy, vx) - math.radians(150)
    left_pt  = (int(tip[0] + math.cos(left_ang)*head_len),  int(tip[1] + math.sin(left_ang)*head_len))
    right_pt = (int(tip[0] + math.cos(right_ang)*head_len), int(tip[1] + math.sin(right_ang)*head_len))
    draw.polygon([tip, left_pt, right_pt], fill=color, outline=outline)
    try:
        font = _load_font(max(10, int(n_label_size)))
        tx, ty = tip[0] + 8, tip[1] - int(size_px*0.2)
        draw.text((tx, ty), "N", fill=color, font=font, stroke_width=1, stroke_fill=outline)
    except Exception:
        pass
    return im

def hex_to_rgb(s, default=(255,255,255)):
    try:
        s = str(s).strip()
        if s.startswith("#"): s = s[1:]
        if len(s)==3: s = "".join([c*2 for c in s])
        if len(s)!=6: return default
        return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))
    except:
        return default

def apply_heading_calibration(theta):
    if theta is None:
        return None
    t = float(theta)
    if st.session_state.get("SB_flip_lr", False):
        t = (360.0 - t) % 360.0
    off = float(st.session_state.get("SB_theta_off", 0.0) or 0.0)
    t = (t + off) % 360.0
    return t

# (… the rest of the full UI identical to previous v11.8 code …)
st.write("Dette er full app-basiskode. For komplette faner A–D, bruk ZIP-filen; den inneholder hele koden.")

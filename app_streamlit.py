
import os, io, re, json, math, zipfile
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import piexif
from pyproj import Transformer, CRS
from streamlit_image_coordinates import streamlit_image_coordinates as img_coords

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

st.set_page_config(page_title="Geotagging bilder v11.4", layout="wide")
st.title("Geotagging bilder v11.4")
st.caption("v11.4 • Punkt/linje < 1 px (0.1–20) – pixel-enheter og minPixels=0 • + alt fra v11.3")

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
        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
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

def draw_north_arrow(im: Image.Image, heading_deg: float,
                     pos=("right","bottom"), size_px=120, margin=20,
                     color=(255,255,255), outline=(0,0,0)) -> Image.Image:
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
    try: draw.text((tip[0]+8, tip[1]-18), "N", fill=color)
    except: pass
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

def detect_columns(df):
    low = {c.lower(): c for c in df.columns}
    col = lambda *names: next((low[n] for n in names if n in low), None)
    return {
        "east": col("øst","oest","east","x"),
        "north": col("nord","north","y"),
        "alt": col("høyde","hoyde","h","z","altitude"),
        "rot": col("rotasjon","retning","dir","heading","azimut","azimuth"),
        "sobj": col("s_objid","sobjid","objid","obj_id","punktid","punkt","id","kum"),
        "hyper": col("s_hyperlink","hyperlink","bilder","filer","photos"),
    }

def dataframe_from_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx",".xls")):
        return pd.read_excel(file, dtype=str)
    else:
        return pd.read_csv(file, dtype=str)

def base_id(s, delims="-_ ./"):
    s=str(s).strip()
    for d in delims:
        if d in s: return s.split(d,1)[0].strip().upper()
    return s.upper()

def _parse_numbers_list(txt):
    if not txt: return []
    parts = str(txt).strip().replace(",", " ").split()
    vals = []
    for p in parts:
        try: vals.append(float(p))
        except: pass
    return vals

def load_lines_landxml(file_obj):
    import xml.etree.ElementTree as ET, io as _io
    text = file_obj.read()
    if isinstance(text, bytes):
        try: text = text.decode("utf-8")
        except: text = text.decode("iso-8859-1", errors="ignore")
    it = ET.iterparse(_io.StringIO(text))
    for _,el in it:
        if '}' in el.tag:
            el.tag = el.tag.split('}',1)[1]
    root = it.root
    lines = []
    for tag in ["PntList2D","PntList3D"]:
        for pl in root.iter():
            if pl.tag.endswith(tag):
                nums = _parse_numbers_list(pl.text or "")
                coords=[]; step=2 if tag=="PntList2D" else 3
                for i in range(0,len(nums)-step+1,step):
                    x=nums[i]; y=nums[i+1]; coords.append((x,y))
                if len(coords)>=2: lines.append({"coords": coords, "objtype": None})
    return lines

def load_lines_geojson(file_obj, prop_objtype=None):
    data = json.load(file_obj)
    feats = data["features"] if "features" in data else [data]
    lines=[]
    for f in feats:
        g = f.get("geometry", {}) or {}
        props = f.get("properties", {}) or {}
        typ = props.get(prop_objtype) if prop_objtype else props.get("objtype") or props.get("type")
        t = g.get("type","")
        if t=="LineString":
            coords=[(float(x),float(y)) for x,y,*_ in g.get("coordinates",[])]
            if len(coords)>=2: lines.append({"coords":coords, "objtype":typ})
        elif t=="MultiLineString":
            for arr in g.get("coordinates", []):
                coords=[(float(x),float(y)) for x,y,*_ in arr]
                if len(coords)>=2: lines.append({"coords":coords, "objtype":typ})
    return lines

def load_lines_auto(file_obj, filename, prop_objtype=None):
    name = filename.lower()
    if name.endswith((".geojson",".json")): return load_lines_geojson(file_obj, prop_objtype)
    if name.endswith((".xml",".landxml")):  return load_lines_landxml(file_obj)
    try:
        file_obj.seek(0); return load_lines_geojson(file_obj, prop_objtype)
    except:
        try: file_obj.seek(0); return load_lines_landxml(file_obj)
        except: return []

def nearest_heading_on_polyline(coords, pt):
    px,py = pt
    best=(None, float("inf"), None)
    for i in range(len(coords)-1):
        x1,y1=coords[i]; x2,y2=coords[i+1]
        vx=x2-x1; vy=y2-y1; L2=vx*vx+vy*vy
        if L2==0: continue
        wx=px-x1; wy=py-y1
        t=(vx*wx+vy*wy)/L2; t=0 if t<0 else (1 if t>1 else t)
        nx=x1+t*vx; ny=y1+t*vy
        dist=((px-nx)**2+(py-ny)**2)**0.5
        if dist<best[1]:
            az=(math.degrees(math.atan2(vx,vy))+360.0)%360.0
            best=(az, dist, (nx,ny))
    return best

with st.sidebar:
    st.header("Prosjektdata (gjelder alle faner)")
    st.subheader("Punkter / Kummer")
    pts_mode = st.radio("Innhold:", ["Hjørner (beregn senter)", "Senterpunkter (direkte E/N)"], index=0, key="SB_pts_mode", horizontal=True)
    pts_up = st.file_uploader("Excel/CSV for ALLE kummer (S_OBJID, Øst, Nord, (Høyde), (Rotasjon))", type=["xlsx","xls","csv"], key="SB_pts")
    epsg_pts = ensure_epsg("SB_epsg_pts", "EPSG for punkter (Øst/Nord)", default=25832)
    delims = st.text_input("Skilletegn for grunn-ID (gruppering)", value="-_ ./", key="SB_delims")
    st.session_state["POINTS_EPSG"] = epsg_pts
    st.session_state["SB_delims"] = delims

    points_df=None
    if pts_up is not None:
        try:
            points_df = dataframe_from_upload(pts_up)
            st.session_state["POINTS_DF"]=points_df
        except Exception as e:
            st.exception(e)

    st.subheader("Linjer (VA/EL)")
    lines_up = st.file_uploader("Linjer (GeoJSON eller XML/LandXML)", type=["geojson","json","xml","landxml"], key="SB_lines")
    epsg_lines = ensure_epsg("SB_epsg_lines", "EPSG for linjer", default=25832)
    st.session_state["LINES_EPSG"]=epsg_lines
    lines_list=None
    if lines_up is not None:
        try:
            lines_list = load_lines_auto(lines_up, lines_up.name)
            st.session_state["LINES_LIST"]=lines_list
        except Exception as e:
            st.exception(e)

tabC, tabD = st.tabs(["C) Manuell + 2-klikk", "D) Kart"])

with tabC:
    st.subheader("Kart – hjørner og linjer (sub‑px)")
    pts_df = st.session_state.get("POINTS_DF")
    lines = st.session_state.get("LINES_LIST")
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    epsg_lin = st.session_state.get("LINES_EPSG", 25832)

    show_corners = st.checkbox("Vis hjørnepunkter", value=True)
    corner_size = st.slider("Størrelse på hjørnepunkter (px)", 0.1, 20.0, 2.0, 0.1)
    line_width = st.slider("Linjebredde (px)", 0.1, 8.0, 0.8, 0.1)

    try:
        import pydeck as pdk
        layers=[]
        # Dummy view
        view_state = pdk.ViewState(latitude=59.91, longitude=10.75, zoom=12)

        if lines:
            if epsg_lin != 4326:
                tr_lin = Transformer.from_crs(epsg_lin, 4326, always_xy=True)
                def to_wgs_path(coords): return [[*tr_lin.transform(x,y)] for (x,y) in coords]
            else:
                def to_wgs_path(coords): return [[x,y] for (x,y) in coords]
            paths = [{"path": to_wgs_path(L["coords"])} for L in lines]
            layers.append(pdk.Layer("PathLayer", paths, get_path="path",
                                    get_width=line_width, width_units="pixels", width_min_pixels=0,
                                    get_color=[80,80,200]))

        if show_corners and pts_df is not None and not pts_df.empty:
            cols = detect_columns(pts_df)
            if cols["east"] and cols["north"]:
                tr_pts = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
                df = pts_df.copy()
                def to_lonlat(e,n):
                    e2=parse_float_maybe_comma(e); n2=parse_float_maybe_comma(n)
                    if e2 is None or n2 is None: return None
                    lo,la = tr_pts.transform(e2,n2)
                    return lo,la
                ll=[to_lonlat(e,n) for e,n in zip(df[cols["east"]], df[cols["north"]])]
                df["lon"]=[p[0] if p else None for p in ll]
                df["lat"]=[p[1] if p else None for p in ll]
                df=df.dropna(subset=["lon","lat"]).reset_index(drop=True)
                df["color"]=[[0,255,0]]*len(df)
                layers.append(pdk.Layer("ScatterplotLayer", df, get_position='[lon, lat]',
                                        get_radius=corner_size, radius_units="pixels", radius_min_pixels=0,
                                        get_fill_color='color', pickable=True))
                if len(df)>0:
                    view_state = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=16)

        st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view_state), use_container_width=True)
    except Exception as e:
        st.info("pydeck ikke tilgjengelig")

with tabD:
    st.subheader("Oversiktskart (sub‑px linjer)")
    line_width_D = st.slider("Linjebredde (px)", 0.1, 8.0, 0.8, 0.1)
    lines = st.session_state.get("LINES_LIST")
    epsg_lin = st.session_state.get("LINES_EPSG", 25832)
    try:
        import pydeck as pdk
        layers=[]
        view_state = pdk.ViewState(latitude=59.91, longitude=10.75, zoom=12)
        if lines:
            if epsg_lin != 4326:
                tr_lin = Transformer.from_crs(epsg_lin, 4326, always_xy=True)
                def to_wgs_path(coords): return [[*tr_lin.transform(x,y)] for (x,y) in coords]
            else:
                def to_wgs_path(coords): return [[x,y] for (x,y) in coords]
            paths = [{"path": to_wgs_path(L["coords"])} for L in lines]
            layers.append(pdk.Layer("PathLayer", paths, get_path="path",
                                    get_width=line_width_D, width_units="pixels", width_min_pixels=0,
                                    get_color=[80,80,200]))
        st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view_state), use_container_width=True)
    except Exception as e:
        st.info("pydeck ikke tilgjengelig")

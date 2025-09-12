
import os, glob, shutil, io, zipfile, re, math, json
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import piexif
from pyproj import Transformer, CRS

st.set_page_config(page_title="Geotagging bilder v10 • Globalt prosjektoppsett • Auto-180 • Linje-heading overalt", layout="wide")

# ===================== Common helpers =====================

def deg_to_dms_rational(dd):
    sign = 1 if dd >= 0 else -1
    dd = abs(dd)
    d = int(dd)
    m_full = (dd - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60 * 10000)
    return sign, ((d,1),(m,1),(s,10000))

def _is_valid_number(x):
    try:
        fx = float(x)
    except Exception:
        return False
    return not (math.isnan(fx) or math.isinf(fx))

def _wrap_deg(d):
    d = float(d) % 360.0
    if d < 0: d += 360.0
    return d

def ang_diff(a, b):
    if a is None or b is None: return None
    d = abs((a - b + 180) % 360 - 180)
    return d

def exif_gps(lat_dd, lon_dd, alt=None, direction=None):
    sign_lat, lat = deg_to_dms_rational(lat_dd)
    sign_lon, lon = deg_to_dms_rational(lon_dd)
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if sign_lat>=0 else b'S',
        piexif.GPSIFD.GPSLatitude: lat,
        piexif.GPSIFD.GPSLongitudeRef: b'E' if sign_lon>=0 else b'W',
        piexif.GPSIFD.GPSLongitude: lon,
    }
    if _is_valid_number(alt):
        alt = float(alt)
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = (int(round(abs(alt)*100)), 100)
    if _is_valid_number(direction):
        direction = _wrap_deg(direction)
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b'T'  # True north
        gps[piexif.GPSIFD.GPSImgDirection] = (int(round(direction*100)), 100)
    return gps

def write_exif(path_in, path_out, lat, lon, alt=None, direction=None):
    im = Image.open(path_in)
    try:
        exif_dict = piexif.load(im.info.get("exif", b""))
    except Exception:
        exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}}
    exif_dict["GPS"] = exif_gps(lat, lon, alt, direction)
    exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
    exif_bytes = piexif.dump(exif_dict)
    im.save(path_out, "jpeg", exif=exif_bytes, quality=95)

def parse_float_maybe_comma(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip().replace(" ", "").replace("\xa0","")
    if s == "" or s.lower() in {"nan","none","-"}: return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except: return None

def ensure_epsg(label_key: str, title: str, default: int = 25832):
    st.markdown(f"**{title}**")
    presets = {
        "EUREF89 / UTM32 (EPSG:25832)": 25832,
        "EUREF89 / UTM33 (EPSG:25833)": 25833,
        "WGS84 (EPSG:4326)": 4326,
        "Custom EPSG (skriv under)": None,
    }
    label = st.selectbox("Velg EPSG (eller Custom):",
                         list(presets.keys()), index=0, key=f"{label_key}_select")
    code = presets[label]
    custom = st.text_input("Custom EPSG (kun tall, f.eks. 5118 for NTM18):",
                           value="", key=f"{label_key}_custom") if code is None else ""
    epsg = code if code is not None else (int(custom) if custom.strip().isdigit() else None)
    if epsg is None:
        st.info(f"Ingen EPSG valgt – bruker default {default}.")
        epsg = default
    try:
        _ = CRS.from_epsg(epsg)
    except Exception:
        st.error(f"Ugyldig EPSG: {epsg}. Bruker default {default}.")
        epsg = default
    return epsg

def transform_EN_to_wgs84(E, N, src_epsg):
    tr = Transformer.from_crs(src_epsg, 4326, always_xy=True)
    lon, lat = tr.transform(float(E), float(N))
    return lat, lon

def transform_EN_to_epsg(E, N, src_epsg, dst_epsg):
    if src_epsg == dst_epsg: return float(E), float(N)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    X, Y = tr.transform(float(E), float(N))
    return X, Y

def normalize_orientation(im: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(im)

def draw_north_arrow(im: Image.Image, heading_deg: float,
                     pos=("right","bottom"), size_px=120, margin=20,
                     color=(255,255,255), outline=(0,0,0)) -> Image.Image:
    if heading_deg is None: return im
    w, h = im.size
    cx = {"left": margin+size_px, "center": w//2, "right": w - margin - size_px}.get(pos[0], w - margin - size_px)
    cy = {"top": margin+size_px, "middle": h//2, "bottom": h - margin - size_px}.get(pos[1], h - margin - size_px)
    ang = math.radians(heading_deg % 360.0)
    dx, dy = 0, -1
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

def sanitize_for_filename(s):
    if s is None: return None
    s = str(s).strip()
    s = re.sub(r'[\\/:*?"<>|]+', '_', s)
    s = re.sub(r'\s+', '_', s)
    s = s.strip('._')
    return s or None

def build_new_name(pattern, label, orig_name, E=None, N=None):
    base, ext = os.path.splitext(orig_name)
    safe_label = sanitize_for_filename(label)
    if pattern == "keep" or not safe_label: return orig_name
    if pattern == "label_orig": return f"{safe_label}_{base}{ext}"
    if pattern == "label_only": return f"{safe_label}{ext}"
    if pattern == "label_en":
        e_txt = f"{int(round(E))}" if E is not None else "E"
        n_txt = f"{int(round(N))}" if N is not None else "N"
        return f"{safe_label}_{e_txt}_{n_txt}{ext}"
    return orig_name

def detect_columns(df):
    low = {c.lower(): c for c in df.columns}
    col = lambda *names: next((low[n] for n in names if n in low), None)
    cols = {
        "east": col("øst","oest","east","x"),
        "north": col("nord","north","y"),
        "alt": col("høyde","hoyde","h","z","altitude"),
        "rot": col("rotasjon","retning","dir","heading","azimut","azimuth"),
        "sobj": col("s_objid","sobjid","objid","obj_id","punktid","punkt","id","kum"),
        "file": col("file","fil","bilde","image","photo","path","sti"),
        "folder": col("folder","mappe","dir","directory"),
        "hyper": col("s_hyperlink","hyperlink","bilder","filer","photos"),
        "epsg": col("epsg","crs"),
        "objtype": col("objtype","type"),
    }
    return cols

def dataframe_from_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx",".xls")):
        return pd.read_excel(file, dtype=str)
    else:
        return pd.read_csv(file, dtype=str)

def expand_hyperlinks(df, col_hyper):
    df2 = df.copy()
    df2[col_hyper] = df2[col_hyper].astype(str).str.split(r"[;|,]")
    df2 = df2.explode(col_hyper).dropna(subset=[col_hyper])
    df2[col_hyper] = df2[col_hyper].str.strip()
    df2 = df2.rename(columns={col_hyper: "file"})
    return df2

def base_id(s, delims="-_ ./"):
    s=str(s).strip()
    for d in delims:
        if d in s: return s.split(d,1)[0].strip().upper()
    m = re.match(r'^([A-ZÆØÅ0-9]+?)(?:\s*(?:HJ(?:ØRNE)?|H|C)?\s*\d+)?$', s.upper())
    return m.group(1) if m else s.upper()

# ====== Lines: GeoJSON + LandXML ======

def _parse_numbers_list(txt):
    if not txt: return []
    parts = txt.strip().replace(",", " ").split()
    vals = []
    for p in parts:
        try: vals.append(float(p))
        except: pass
    return vals

def load_lines_landxml(file_obj):
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_obj); root = tree.getroot()
    ns = {}
    if root.tag.startswith("{"):
        uri = root.tag.split("}",1)[0][1:]
        ns['lx'] = uri
    pnt_by_name = {}
    for p in root.iter():
        if p.tag.endswith("Pnt") and p.get("name"):
            nums = _parse_numbers_list(p.text or "")
            if len(nums) >= 2:
                pnt_by_name[p.get("name")] = (nums[0], nums[1])
    lines = []
    for tag in ["PntList2D","PntList3D"]:
        for pl in root.iter():
            if pl.tag.endswith(tag):
                nums = _parse_numbers_list(pl.text or "")
                coords=[]; step=2 if tag=="PntList2D" else 3
                for i in range(0, len(nums)-step+1, step):
                    x=nums[i]; y=nums[i+1]; coords.append((x,y))
                if len(coords)>=2: lines.append({"coords": coords, "objtype": None})
    for al in root.iter():
        if not al.tag.endswith("Alignment"): continue
        for cg in al.iter():
            if not cg.tag.endswith("CoordGeom"): continue
            seg=[]
            def get_xy(node):
                if node is None: return None
                ref = node.get("pntRef") if hasattr(node,"get") else None
                if ref and ref in pnt_by_name: return pnt_by_name[ref]
                nums = _parse_numbers_list(getattr(node,"text",None) or "")
                if len(nums)>=2: return (nums[0], nums[1])
                try:
                    x=float(node.get("x")); y=float(node.get("y")); return (x,y)
                except: return None
            for geom in list(cg):
                tag = geom.tag.split("}")[-1]
                if tag in ("Line","Curve","Spiral"):
                    st = en = None
                    for child in list(geom):
                        ctag = child.tag.split("}")[-1]
                        if ctag.lower().startswith("start"): st = get_xy(child)
                        elif ctag.lower().startswith("end"): en = get_xy(child)
                    if st and (not seg or seg[-1]!=st): seg.append(st)
                    if en: seg.append(en)
            if len(seg)>=2: lines.append({"coords": seg, "objtype": None})
    return lines

def load_lines_geojson(file_obj, prop_objtype=None):
    data = json.load(file_obj)
    feats = data["features"] if "features" in data else [data]
    lines = []
    for f in feats:
        g = f.get("geometry", {}) or {}
        props = f.get("properties",{}) or {}
        typ = props.get(prop_objtype) if prop_objtype else props.get("objtype") or props.get("type")
        t = g.get("type","")
        if t == "LineString":
            coords = [(float(x),float(y)) for x,y,*_ in g.get("coordinates",[])]
            if len(coords)>=2: lines.append({"coords": coords, "objtype": typ})
        elif t == "MultiLineString":
            for arr in g.get("coordinates", []):
                coords = [(float(x),float(y)) for x,y,*_ in arr]
                if len(coords)>=2: lines.append({"coords": coords, "objtype": typ})
    return lines

def load_lines_auto(file_obj, filename, prop_objtype=None):
    name = filename.lower()
    if name.endswith((".geojson",".json")):
        return load_lines_geojson(file_obj, prop_objtype=prop_objtype)
    if name.endswith((".xml",".landxml")):
        return load_lines_landxml(file_obj)
    try:
        file_obj.seek(0); return load_lines_geojson(file_obj, prop_objtype=prop_objtype)
    except Exception:
        try:
            file_obj.seek(0); return load_lines_landxml(file_obj)
        except Exception:
            return []

def nearest_heading_on_polyline(coords, pt):
    px, py = pt
    best = (None, float("inf"), None)
    for i in range(len(coords)-1):
        x1,y1 = coords[i]; x2,y2 = coords[i+1]
        vx = x2-x1; vy = y2-y1; L2 = vx*vx + vy*vy
        if L2 == 0: continue
        wx = px-x1; wy = py-y1
        t = (vx*wx + vy*wy) / L2
        t = 0 if t<0 else (1 if t>1 else t)
        nx = x1 + t*vx; ny = y1 + t*vy
        dist = ((px-nx)**2 + (py-ny)**2)**0.5
        if dist < best[1]:
            az = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0  # 0°=N
            best = (az, dist, (nx,ny))
    return best

# ===================== Sidebar: Global Project Data =====================

st.title("Geotagging bilder v10")
st.caption("Globalt prosjektoppsett i sidepanelet • Linje-heading i alle faner • Auto-180 • S_HYPERLINK • Kum-senter • Kart og korrektur")

with st.sidebar:
    st.header("Prosjektdata (gjelder alle faner)")

    # Points file
    st.subheader("Punkter / Kummer")
    pts_mode = st.radio("Innhold:", ["Hjørner (beregn senter)", "Senterpunkter (direkte E/N)"], index=0, key="SB_pts_mode", horizontal=True)
    pts_up = st.file_uploader("Excel/CSV for ALLE kummer (S_OBJID, Øst, Nord, (Høyde), (Rotasjon))", type=["xlsx","xls","csv"], key="SB_pts")
    epsg_pts = ensure_epsg("SB_epsg_pts", "EPSG for punkter (Øst/Nord)", default=25832)
    delims = st.text_input("Skilletegn for grunn-ID (gruppering)", value="-_ ./", key="SB_delims")

    points_df = None
    centers_dict = None
    centers_df = None

    if pts_up is not None:
        try:
            points_df = dataframe_from_upload(pts_up)
            cols = detect_columns(points_df)
            if not cols["east"] or not cols["north"]:
                st.error("Fant ikke Øst/Nord i punktfilen.")
            else:
                if pts_mode.startswith("Hjørner"):
                    if not cols["sobj"]:
                        st.error("Trenger S_OBJID for å gruppere hjørner.")
                    else:
                        import numpy as np
                        def pca_heading(points):
                            A=np.array(points, dtype=float)
                            c=A.mean(axis=0); A0=A-c
                            U,S,Vt=np.linalg.svd(A0, full_matrices=False)
                            vx,vy=Vt[0,0],Vt[0,1]
                            if vy<0: vx,vy=-vx,-vy
                            az=(math.degrees(math.atan2(vx,vy))+360.0)%360.0
                            proj_main=A0@Vt[0,:]; proj_ortho=A0@Vt[1,:]
                            L=(proj_main.max()-proj_main.min()); W=(proj_ortho.max()-proj_ortho.min())
                            return (float(c[0]),float(c[1])), az, abs(float(L)), abs(float(W))
                        points_df["_base"] = points_df[cols["sobj"]].astype(str).map(lambda s: base_id(s, delims))
                        recs=[]
                        for base, grp in points_df.groupby("_base"):
                            pts=[]; zs=[]
                            for _,r in grp.iterrows():
                                e=parse_float_maybe_comma(r[cols["east"]]); n=parse_float_maybe_comma(r[cols["north"]])
                                if e is None or n is None: continue
                                pts.append((e,n))
                                if cols["alt"]:
                                    z=parse_float_maybe_comma(r[cols["alt"]]); 
                                    if z is not None: zs.append(z)
                            if len(pts)>=3:
                                (ce,cn), az, L, W = pca_heading(pts)
                                zmed=float(pd.Series(zs).median()) if len(zs)>0 else None
                                recs.append({"base_id":base,"center_E":ce,"center_N":cn,"azimut":az,"hoyde":zmed,"count":len(pts)})
                        if recs:
                            centers_df = pd.DataFrame(recs).sort_values("base_id")
                            st.success(f"Kum-senter beregnet for {len(centers_df)} kummer.")
                            st.dataframe(centers_df.head(30), use_container_width=True)
                            centers_dict = {r["base_id"]: r for _, r in centers_df.iterrows()}
                else:
                    # direct centers
                    show_cols = [c for c in [cols["sobj"], cols["east"], cols["north"], cols["alt"], cols["rot"]] if c]
                    st.success(f"Fant {len(points_df)} senterpunkter.")
                    st.dataframe(points_df[show_cols].head(30), use_container_width=True)
                    # fabricate centers_dict
                    centers_dict = {}
                    if cols["sobj"]:
                        for _, r in points_df.iterrows():
                            sobj = str(r[cols["sobj"]]).strip()
                            if sobj:
                                centers_dict[base_id(sobj, delims)] = {
                                    "base_id": base_id(sobj, delims),
                                    "center_E": parse_float_maybe_comma(r[cols["east"]]),
                                    "center_N": parse_float_maybe_comma(r[cols["north"]]),
                                    "azimut": parse_float_maybe_comma(r[cols["rot"]]) if cols["rot"] else None,
                                    "hoyde": parse_float_maybe_comma(r[cols["alt"]]) if cols["alt"] else None,
                                }
        except Exception as e:
            st.exception(e)

    # Lines
    st.subheader("Linjer (VA/EL)")
    lines_up = st.file_uploader("Linjer (GeoJSON eller XML/LandXML)", type=["geojson","json","xml","landxml"], key="SB_lines")
    objtype_field = st.text_input("Objekttype-felt i linjefil (valgfritt)", value="objtype", key="SB_objfield")
    epsg_lines = ensure_epsg("SB_epsg_lines", "EPSG for linjer", default=25832)
    type_filter = st.text_input("Typefilter (komma-separert, valgfritt)", value="", key="SB_typefilter")
    buffer_m = st.number_input("Buffer (m) mot linje", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="SB_buffer")

    lines_list = None
    if lines_up is not None:
        try:
            lines_list = load_lines_auto(lines_up, lines_up.name, prop_objtype=objtype_field or None)
            if type_filter.strip():
                allowed = set([s.strip() for s in type_filter.split(",") if s.strip()])
                lines_list = [L for L in lines_list if (L["objtype"] in allowed)]
        except Exception as e:
            st.exception(e)

    # Global options
    st.subheader("Globale valg")
    draw_arrow_global = st.checkbox("Tegn nordpil på bilder", value=True, key="SB_draw_arrow")
    arrow_size_global = st.slider("Pil-størrelse (px)", 60, 240, 120, key="SB_arrow_size")
    auto_180 = st.checkbox("Auto-180 (flipp heading hvis ~180° fra senter/rotasjon)", value=True, key="SB_auto180")

    # Save to session
    st.session_state["POINTS_EPSG"] = epsg_pts
    st.session_state["LINES_EPSG"] = epsg_lines
    st.session_state["BUFFER_M"] = buffer_m
    st.session_state["DRAW_ARROW"] = draw_arrow_global
    st.session_state["ARROW_SIZE"] = arrow_size_global
    st.session_state["AUTO_180"] = auto_180
    st.session_state["CENTERS_DICT"] = centers_dict
    st.session_state["CENTERS_DF"] = centers_df
    st.session_state["POINTS_DF"] = points_df
    st.session_state["LINES_LIST"] = lines_list

# ===================== Heading & position chooser =====================

def heading_from_lines(E, N):
    lines = st.session_state.get("LINES_LIST")
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    epsg_lin = st.session_state.get("LINES_EPSG", 25832)
    buf = st.session_state.get("BUFFER_M", 2.0)
    if not lines: 
        return None, None
    # Reproject lines to point CRS if needed (do a shallow copy to avoid mutating original repeatedly)
    if epsg_lin != epsg_pts:
        tr = Transformer.from_crs(epsg_lin, epsg_pts, always_xy=True)
        def reproject_coords(coords):
            return [tuple(tr.transform(x,y)) for (x,y) in coords]
        lines_use = [{"coords": reproject_coords(L["coords"]), "objtype": L.get("objtype")} for L in lines]
    else:
        lines_use = lines
    best = (None, float("inf"))
    for L in lines_use:
        hd, dist, npt = nearest_heading_on_polyline(L["coords"], (E,N))
        if npt is None: 
            continue
        if dist < best[1]:
            best = (hd, dist)
    hd, dist = best
    if dist is None: return None, None
    return (hd if dist <= buf else None), dist

def choose_pos_and_heading(sobj_label=None, E=None, N=None, Alt=None, Rot=None):
    # Position priority: provided E/N -> center of sobj
    centers = st.session_state.get("CENTERS_DICT") or {}
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)

    center_hint = None
    if (E is None or N is None) and sobj_label:
        base = base_id(sobj_label)
        info = centers.get(base)
        if info:
            E = info.get("center_E"); N = info.get("center_N"); 
            if Alt is None: Alt = info.get("hoyde")
            center_hint = info.get("azimut")

    # Heading priority: line heading -> center azimut -> Rot
    line_h, dist = heading_from_lines(E, N) if (E is not None and N is not None) else (None, None)
    hd = line_h if _is_valid_number(line_h) else (center_hint if _is_valid_number(center_hint) else (Rot if _is_valid_number(Rot) else None))

    # Auto-180 check
    if st.session_state.get("AUTO_180", True):
        # if both line_h and center_hint are present and ~180 apart, flip line
        if _is_valid_number(line_h) and _is_valid_number(center_hint):
            d = ang_diff(line_h, center_hint)
            if d is not None and 150 <= d <= 210:
                hd = (line_h + 180) % 360
        # else if line missing, but Rot and center exist and ~180, flip Rot
        elif not _is_valid_number(line_h) and _is_valid_number(Rot) and _is_valid_number(center_hint):
            d = ang_diff(Rot, center_hint)
            if d is not None and 150 <= d <= 210:
                hd = (Rot + 180) % 360

    return E, N, Alt, (_wrap_deg(hd) if _is_valid_number(hd) else None), center_hint, line_h, dist

# ===================== Tabs =====================

tabA, tabB, tabD = st.tabs(["A) Batch geotagg", "B) CSV-mapping", "D) Korrektur + kart"])

# ---------- Tab A: multi images with global project data ----------
with tabA:
    st.subheader("A) Geotagg mange bilder (bruk prosjektdata fra sidepanelet)")
    # Pick a kum/S_OBJID from centers
    centers_df = st.session_state.get("CENTERS_DF")
    centers_dict = st.session_state.get("CENTERS_DICT") or {}
    picked_label = None
    if centers_dict:
        options = sorted(list(centers_dict.keys()))
        picked_label = st.selectbox("Velg kum/S_OBJID (for å bruke kum-senter + heading-hint):", options, key="A_pick_label")
    else:
        st.info("Ingen kum-senter tilgjengelig ennå. Last opp punkter i sidepanelet.")

    mode = st.radio("Bildekilde:", ["Lokal mappe", "ZIP-opplasting", "Opplasting (flere filer)"], index=2, key="A_mode")
    colA1, colA2 = st.columns(2)
    with colA1:
        if mode == "Lokal mappe":
            rootA = st.text_input("Aktiv mappe (lokal sti)", value="", key="A_root")
            outA = st.text_input("Ut-mappe (kopi; tom = overskriv i aktiv mappe)", value="", key="A_out")
            overwriteA = st.checkbox("Overskriv originaler (forsiktig)", value=False, key="A_overwrite")
        elif mode == "ZIP-opplasting":
            zip_up = st.file_uploader("Last opp ZIP med JPG-bilder", type=["zip"], key="A_zip")
        else:
            files_up = st.file_uploader("Dra og slipp mange JPG", type=["jpg","jpeg"], accept_multiple_files=True, key="A_files")

        rename_pattern_A = st.selectbox("Nytt filnavn (mønster)", [
            "Behold originalt navn","S_OBJID + originalt navn","Kun S_OBJID","S_OBJID + avrundet E/N"
        ], index=1, key="A_rename")

    with colA2:
        draw_arrow = st.session_state.get("DRAW_ARROW", True)
        arrow_size = st.session_state.get("ARROW_SIZE", 120)
        epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
        epsg_out = ensure_epsg("TAB_A_DOC_EPSG", "Dokumentasjons-CRS for CSV (eksport)", default=25832)

    patt_map = {"Behold originalt navn":"keep","S_OBJID + originalt navn":"label_orig","Kun S_OBJID":"label_only","S_OBJID + avrundet E/N":"label_en"}
    patt = patt_map[rename_pattern_A]

    if st.button("Kjør geotag (Tab A)", key="A_run"):
        try:
            rows_out = []
            if mode == "Lokal mappe":
                if not rootA or not os.path.isdir(rootA): st.error("Aktiv mappe finnes ikke.")
                else:
                    out_dir = rootA if (overwriteA or not outA) else outA
                    if not overwriteA and outA: os.makedirs(outA, exist_ok=True)
                    jpgs = sorted(glob.glob(os.path.join(rootA, "*.jpg")) + glob.glob(os.path.join(rootA, "*.JPG")))
                    for p in jpgs:
                        fname=os.path.basename(p)
                        E=N=Alt=Rot=None
                        sobj = picked_label or None
                        E,N,Alt,hd,cent_hint,line_h,dist = choose_pos_and_heading(sobj, E,N, Alt, Rot)
                        if E is None or N is None:
                            st.warning(f"Hopper {fname} (mangler posisjon)."); continue
                        lat,lon = transform_EN_to_wgs84(E,N, epsg_pts)
                        newname = build_new_name(patt, sobj, fname, E, N)
                        dst = p if (overwriteA and not outA and newname==fname) else os.path.join(out_dir, newname)
                        if dst != p:
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.copy2(p, dst)
                        im=Image.open(dst); im=normalize_orientation(im)
                        if draw_arrow and _is_valid_number(hd): im=draw_north_arrow(im, _wrap_deg(hd), size_px=arrow_size)
                        im.save(dst, "jpeg", quality=95)
                        write_exif(dst, dst, lat, lon, Alt, hd)
                        Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_pts, epsg_out)
                        rows_out.append({
                            "file": os.path.basename(dst), "S_OBJID": sobj, "E_in": E, "N_in": N, "lat": lat, "lon": lon,
                            f"E_{epsg_out}": Xdoc, f"N_{epsg_out}": Ydoc, "hoyde": Alt, "heading": hd,
                            "heading_line": line_h, "center_hint": cent_hint, "dist_to_line": dist
                        })
            elif mode == "ZIP-opplasting":
                if not zip_up: st.error("Last opp ZIP.")
                else:
                    zin = zipfile.ZipFile(io.BytesIO(zip_up.read()), "r")
                    zout_mem = io.BytesIO(); zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)
                    used=set()
                    for name in zin.namelist():
                        if not name.lower().endswith((".jpg",".jpeg")): continue
                        sobj = picked_label or None
                        E=N=Alt=Rot=None
                        E,N,Alt,hd,cent_hint,line_h,dist = choose_pos_and_heading(sobj, E,N, Alt, Rot)
                        if E is None or N is None:
                            continue
                        lat,lon = transform_EN_to_wgs84(E,N, epsg_pts)
                        payload = zin.read(name)
                        im = Image.open(io.BytesIO(payload)); im=normalize_orientation(im)
                        if draw_arrow and _is_valid_number(hd): im=draw_north_arrow(im, _wrap_deg(hd), size_px=arrow_size)
                        out_io=io.BytesIO(); im.save(out_io, "jpeg", quality=95); out_io.seek(0)
                        tmp=out_io.getvalue(); open("tmp.jpg","wb").write(tmp)
                        write_exif("tmp.jpg", "tmp.jpg", lat, lon, Alt, hd)
                        final=open("tmp.jpg","rb").read()
                        newname=build_new_name(patt, sobj, os.path.basename(name), E, N)
                        base,ext=os.path.splitext(newname); i=1; cand=newname
                        while cand in used:
                            cand=f"{base}_{i}{ext}"; i+=1
                        used.add(cand)
                        zout.writestr(cand, final)
                        Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_pts, epsg_out)
                        rows_out.append({"file": cand, "S_OBJID": sobj, "E_in": E, "N_in": N, "lat": lat, "lon": lon,
                                         f"E_{epsg_out}": Xdoc, f"N_{epsg_out}": Ydoc, "hoyde": Alt, "heading": hd,
                                         "heading_line": line_h, "center_hint": cent_hint, "dist_to_line": dist})
                    zout.close(); st.download_button("Last ned geotagget ZIP", data=zout_mem.getvalue(), file_name="geotagged.zip", mime="application/zip")
            else:
                files_up = st.session_state.get("A_files") or st.session_state.get("A_files_dummy")
                # read directly from uploader widget
                files_up = st.session_state.get("uploaded_files", None)
                # Simpler: use the widget variable
                files_up = st.session_state.get("A_files")
                # but Streamlit stores in the widget; we access through key provided in uploader. We'll handle inline below:

            # For the simple case (multiple files upload) handle inline due to Streamlit limitations:
            if st.session_state.get("A_mode") == "Opplasting (flere filer)":
                files_up_w = st.session_state.get("A_files")
                if files_up_w:
                    zout_mem=io.BytesIO(); zout=zipfile.ZipFile(zout_mem,"w",zipfile.ZIP_DEFLATED)
                    used=set()
                    for f in files_up_w:
                        name = f.name
                        sobj = picked_label or None
                        E=N=Alt=Rot=None
                        E,N,Alt,hd,cent_hint,line_h,dist = choose_pos_and_heading(sobj, E,N, Alt, Rot)
                        if E is None or N is None: 
                            continue
                        lat, lon = transform_EN_to_wgs84(E,N, epsg_pts)
                        im=Image.open(f); im=normalize_orientation(im)
                        if draw_arrow and _is_valid_number(hd): im=draw_north_arrow(im, _wrap_deg(hd), size_px=arrow_size)
                        out_io=io.BytesIO(); im.save(out_io, "jpeg", quality=95); out_io.seek(0)
                        tmp=out_io.getvalue(); open("tmp.jpg","wb").write(tmp)
                        write_exif("tmp.jpg", "tmp.jpg", lat, lon, Alt, hd)
                        final=open("tmp.jpg","rb").read()
                        newname=build_new_name(patt, sobj, os.path.basename(name), E, N)
                        base,ext=os.path.splitext(newname); i=1; cand=newname
                        while cand in used:
                            cand=f"{base}_{i}{ext}"; i+=1
                        used.add(cand)
                        zout.writestr(cand, final)
                        Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_pts, epsg_out)
                        rows_out.append({"file": cand, "S_OBJID": sobj, "E_in": E, "N_in": N, "lat": lat, "lon": lon,
                                         f"E_{epsg_out}": Xdoc, f"N_{epsg_out}": Ydoc, "hoyde": Alt, "heading": hd,
                                         "heading_line": line_h, "center_hint": cent_hint, "dist_to_line": dist})
                    zout.close(); st.download_button("Last ned ZIP med geotaggede bilder", data=zout_mem.getvalue(), file_name="geotagged_upload.zip", mime="application/zip")

            if rows_out:
                dfo=pd.DataFrame(rows_out)
                st.success(f"Geotagget {len(dfo)} bilder.")
                st.download_button("Last ned CSV (Tab A)", dfo.to_csv(index=False).encode("utf-8"), "geotag_tabA.csv", "text/csv")
        except Exception as e:
            st.exception(e)

# ---------- Tab B: CSV-mapping (uses same global project data + arrow) ----------
with tabB:
    st.subheader("B) CSV/Excel-mapping (inkl. S_HYPERLINK) – bruker linjer/senter fra sidepanelet")
    root = st.text_input("Root-mappe for bilder (lokal sti)", value="", key="B_root")
    up = st.file_uploader("Mapping-CSV/Excel (kan ha S_HYPERLINK)", type=["csv","xlsx","xls"], key="B_csv")
    out_root = st.text_input("Ut-root (kopi; tom = overskriv)", value="", key="B_out")
    overwrite2 = st.checkbox("Overskriv originale", value=False, key="B_overwrite")
    rename_pattern_B = st.selectbox("Nytt filnavn", [
        "Behold originalt navn","S_OBJID + originalt navn","Kun S_OBJID","S_OBJID + avrundet E/N"
    ], index=1, key="B_rename")
    draw_arrow_B = st.checkbox("Tegn nordpil (bruker globale innstillinger)", value=True, key="B_arrow")
    arrow_size_B = st.session_state.get("ARROW_SIZE", 120)
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    epsg_out_doc2 = ensure_epsg("TAB_B_DOC_EPSG", "Dokumentasjons-CRS for CSV (kun eksport)", default=25832)

    runB = st.button("Kjør geotag (Tab B)", key="B_run")
    if runB:
        try:
            if not root or not os.path.isdir(root):
                st.error("Root finnes ikke.")
            elif not up:
                st.error("Last opp CSV/Excel.")
            else:
                dfm = dataframe_from_upload(up)
                cols = detect_columns(dfm)
                if cols["hyper"] and cols["hyper"] in dfm.columns:
                    dfm = expand_hyperlinks(dfm, cols["hyper"]); cols["file"] = "file"
                if not cols["file"]:
                    st.error("Mangler 'file' eller S_HYPERLINK.")
                else:
                    rows = []
                    patt_map = {"Behold originalt navn":"keep","S_OBJID + originalt navn":"label_orig","Kun S_OBJID":"label_only","S_OBJID + avrundet E/N":"label_en"}
                    patt = patt_map[rename_pattern_B]
                    for _, r in dfm.dropna(subset=[cols["file"]]).iterrows():
                        sobj = str(r[cols["sobj"]]) if cols["sobj"] else None
                        # take E/N if present in mapping, else from center for sobj
                        E = parse_float_maybe_comma(r[cols["east"]]) if cols["east"] in dfm.columns else None
                        N = parse_float_maybe_comma(r[cols["north"]]) if cols["north"] in dfm.columns else None
                        Alt = parse_float_maybe_comma(r[cols["alt"]]) if cols["alt"] in dfm.columns else None
                        Rot = parse_float_maybe_comma(r[cols["rot"]]) if cols["rot"] in dfm.columns else None
                        E,N,Alt,hd,cent_hint,line_h,dist = choose_pos_and_heading(sobj, E,N, Alt, Rot)
                        if E is None or N is None: 
                            st.warning(f"Hopper '{r[cols['file']]}' (mangler posisjon)."); continue
                        p = os.path.join(root, str(r[cols["file"]]))
                        if not os.path.isfile(p): st.warning(f"Mangler fil: {p}"); continue
                        lat, lon = transform_EN_to_wgs84(E,N, epsg_pts)
                        newname = build_new_name(patt, sobj, os.path.basename(p), E, N)
                        dst = p if overwrite2 else os.path.join(out_root or os.path.dirname(p), newname)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if dst != p: shutil.copy2(p, dst)
                        im = Image.open(dst); im = normalize_orientation(im)
                        if draw_arrow_B and st.session_state.get("DRAW_ARROW", True) and _is_valid_number(hd):
                            im = draw_north_arrow(im, _wrap_deg(hd), size_px=arrow_size_B)
                        im.save(dst, "jpeg", quality=95)
                        write_exif(dst, dst, lat, lon, Alt, hd)
                        Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_pts, epsg_out_doc2)
                        rows.append({"file": os.path.relpath(dst, out_root) if out_root else os.path.basename(dst),
                                     "S_OBJID": sobj, "E_in": E, "N_in": N, "lat": lat, "lon": lon,
                                     f"E_{epsg_out_doc2}": Xdoc, f"N_{epsg_out_doc2}": Ydoc,
                                     "hoyde": Alt, "heading": hd, "heading_line": line_h, "center_hint": cent_hint, "dist_to_line": dist})
                    dfr=pd.DataFrame(rows); st.success(f"Geotagget {len(dfr)} bilder.")
                    st.download_button("Last ned CSV (Tab B)", dfr.to_csv(index=False).encode("utf-8"), "geotag_tabB.csv", "text/csv")
        except Exception as e:
            st.exception(e)

# ---------- Tab D: correction + map (uses global table produced by A/B) ----------
with tabD:
    st.subheader("D) Korrektur + kart")
    st.markdown("Kjør først A/B for å generere en tabell i minnet – deretter kan du korrigere heading og skrive EXIF på nytt.")

    if "LAST_RUN_TABLE" not in st.session_state:
        st.session_state["LAST_RUN_TABLE"] = None

    # If A/B just ran, store/refresh table
    # (We add a small button for 'Hent siste resultat fra Tab A/B' – user clicks after A/B)
    if st.button("Hent siste resultat fra Tab A/B", key="D_get"):
        # No direct pipeline here; A/B already produce CSV download. For in-memory, we skip.
        st.info("I denne lette versjonen lastes ikke tabellen automatisk. (Kan utvides til full minnelagring ved behov.)")

    # Minimal map from current centers + lines to validate project data:
    try:
        import pydeck as pdk
        centers_df = st.session_state.get("CENTERS_DF")
        lines = st.session_state.get("LINES_LIST")
        epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
        epsg_lin = st.session_state.get("LINES_EPSG", 25832)
        layers = []
        if centers_df is not None and not centers_df.empty:
            tr_pts = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
            tmp = centers_df.copy()
            tmp["lon"], tmp["lat"] = zip(*[tr_pts.transform(e, n) for e,n in zip(tmp["center_E"], tmp["center_N"])])
            tmp["color"] = [ [0,150,255] ] * len(tmp)
            layers.append(pdk.Layer("ScatterplotLayer", tmp, get_position='[lon, lat]', get_radius=4, get_fill_color='color', pickable=True))
            view_state = pdk.ViewState(latitude=float(tmp["lat"].mean()), longitude=float(tmp["lon"].mean()), zoom=16)
        else:
            view_state = pdk.ViewState(latitude=59.91, longitude=10.75, zoom=10)

        if lines:
            if epsg_lin != 4326:
                tr_lin = Transformer.from_crs(epsg_lin, 4326, always_xy=True)
                def to_wgs_path(coords): return [[*tr_lin.transform(x,y)][::-1][::-1] and [tr_lin.transform(x,y)[0], tr_lin.transform(x,y)[1]] for (x,y) in coords]
            else:
                def to_wgs_path(coords): return [[x,y] for (x,y) in coords]
            paths = [{"path": to_wgs_path(L["coords"])} for L in lines]
            layers.append(pdk.Layer("PathLayer", paths, get_path="path", get_width=2, get_color=[80,80,200]))
        st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view_state), use_container_width=True)
    except Exception as e:
        st.info("Kunne ikke vise kart (pydeck mangler eller feil).")

st.markdown("---")
st.caption("v10 • Global sidebar (punkter/linjer) • heading fra linjer overalt • auto-180 • S_OBJID-prefiks • EXIF WGS84 • nordpil")

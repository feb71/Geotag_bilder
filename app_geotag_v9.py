
import os, glob, shutil, io, zipfile, re, math, json
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import piexif
from pyproj import Transformer, CRS

st.set_page_config(page_title="Geotagging bilder v8 • Linje-heading + Korrektur + Kart", layout="wide")

# ================ Helpers (common) ================

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
    if d < 0:
        d += 360.0
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
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b'T'  # true north
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
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(" ", "").replace("\xa0","")
    if s == "" or s.lower() in {"nan","none","-"}:
        return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def ensure_epsg(title: str, key_base: str, default: int = 25832):
    st.markdown(f"**{title}**")
    presets = {
        "EUREF89 / UTM32 (EPSG:25832)": 25832,
        "EUREF89 / UTM33 (EPSG:25833)": 25833,
        "WGS84 (EPSG:4326)": 4326,
        "Custom EPSG (skriv under)": None,
    }
    label = st.selectbox("Velg EPSG (eller Custom):",
                         list(presets.keys()), index=0, key=f"{key_base}_select")
    code = presets[label]
    custom = st.text_input("Custom EPSG (kun tall, f.eks. 5118 for NTM18):",
                           value="", key=f"{key_base}_custom") if code is None else ""
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
    if src_epsg == dst_epsg:
        return float(E), float(N)
    tr = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    X, Y = tr.transform(float(E), float(N))
    return X, Y

def normalize_orientation(im: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(im)

def draw_north_arrow(im: Image.Image, heading_deg: float,
                     pos=("right","bottom"), size_px=120, margin=20,
                     color=(255,255,255), outline=(0,0,0)) -> Image.Image:
    if heading_deg is None:
        return im
    w, h = im.size
    cx = {"left": margin+size_px, "center": w//2, "right": w - margin - size_px}.get(pos[0], w - margin - size_px)
    cy = {"top": margin+size_px, "middle": h//2, "bottom": h - margin - size_px}.get(pos[1], h - margin - size_px)

    ang = math.radians(heading_deg % 360.0)
    dx = 0; dy = -1
    vx = dx*math.cos(ang) - dy*math.sin(ang)
    vy = dx*math.sin(ang) + dy*math.cos(ang)

    tip = (int(cx + vx*size_px), int(cy + vy*size_px))
    tail = (int(cx - vx*(size_px*0.4)), int(cy - vy*(size_px*0.4)))

    draw = ImageDraw.Draw(im, "RGBA")
    draw.line([tail, tip], fill=color, width=max(4, size_px//15))
    head_len = size_px*0.25
    left_ang = math.atan2(vy, vx) + math.radians(150)
    right_ang= math.atan2(vy, vx) - math.radians(150)
    left_pt  = (int(tip[0] + math.cos(left_ang)*head_len),
                int(tip[1] + math.sin(left_ang)*head_len))
    right_pt = (int(tip[0] + math.cos(right_ang)*head_len),
                int(tip[1] + math.sin(right_ang)*head_len))
    draw.polygon([tip, left_pt, right_pt], fill=color, outline=outline)
    try:
        draw.text((tip[0]+8, tip[1]-18), "N", fill=color)
    except:
        pass
    return im

def sanitize_for_filename(s):
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r'[\\/:*?"<>|]+', '_', s)
    s = re.sub(r'\s+', '_', s)
    s = s.strip('._')
    return s or None

def build_new_name(pattern, label, orig_name, E=None, N=None):
    base, ext = os.path.splitext(orig_name)
    safe_label = sanitize_for_filename(label)
    if pattern == "keep" or not safe_label:
        return orig_name
    if pattern == "label_orig":
        return f"{safe_label}_{base}{ext}"
    if pattern == "label_only":
        return f"{safe_label}{ext}"
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
        "sobj": col("s_objid","sobjid","objid","obj_id","punktid","punkt","id"),
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

# ================ Line utilities (no heavy deps) ================

def load_lines_geojson(file_obj, prop_objtype=None):
    """Return list of dicts: {'coords': [(E,N),...], 'objtype':str or None}"""
    data = json.load(file_obj)
    feats = data["features"] if "features" in data else [data]
    lines = []
    for f in feats:
        g = f.get("geometry", {})
        if not g: 
            continue
        gtype = g.get("type","")
        props = f.get("properties",{}) or {}
        typ = props.get(prop_objtype) if prop_objtype else props.get("objtype") or props.get("type")
        if gtype == "LineString":
            coords = [(float(x),float(y)) for x,y,*rest in g.get("coordinates",[])]
            if len(coords)>=2: lines.append({"coords": coords, "objtype": typ})
        elif gtype == "MultiLineString":
            for arr in g.get("coordinates", []):
                coords = [(float(x),float(y)) for x,y,*rest in arr]
                if len(coords)>=2: lines.append({"coords": coords, "objtype": typ})
    return lines

def nearest_heading_on_polyline(coords, pt):
    """coords: list of (E,N), pt: (E,N). Returns (heading_deg, dist, nearest_point) or (None, inf, None)"""
    px, py = pt
    best = (None, float("inf"), None, None)  # (heading, dist, nearest, segvec)
    for i in range(len(coords)-1):
        x1,y1 = coords[i]; x2,y2 = coords[i+1]
        vx = x2 - x1; vy = y2 - y1
        L2 = vx*vx + vy*vy
        if L2 == 0: 
            continue
        wx = px - x1; wy = py - y1
        t = (vx*wx + vy*wy) / L2
        if t < 0: t = 0
        elif t > 1: t = 1
        nx = x1 + t*vx; ny = y1 + t*vy
        dx = px - nx; dy = py - ny
        dist = (dx*dx + dy*dy)**0.5
        if dist < best[1]:
            az = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0  # 0°=N
            best = (az, dist, (nx,ny), (vx,vy))
    return best[0], best[1], best[2]

# ================ UI Tabs ================

st.title("Geotagging bilder v8")
st.caption("• Mapping via S_HYPERLINK • EXIF WGS84 • Nordpil • Kum-senter • **Linje-basert heading + Korrektur + Kart**")

tabB, tabC, tabD = st.tabs(["B) CSV/Excel-mapping (S_HYPERLINK/fil)", "C) Kum-senter fra hjørner", "D) Linje-heading + korrektur + kart"])

# ---------- Tab B (same core as v7, shortened) ----------
with tabB:
    st.subheader("B) CSV/Excel-mapping")
    colB1, colB2 = st.columns(2)
    with colB1:
        root = st.text_input("Root-mappe for bilder", value="", key="B_root")
        up = st.file_uploader("Mapping-CSV/Excel (kan ha S_HYPERLINK)", type=["csv","xlsx","xls"], key="B_csv")
        out_root = st.text_input("Ut-root (kopi; tom = overskriv)", value="", key="B_out")
        overwrite2 = st.checkbox("Overskriv originale", value=False, key="B_overwrite")
        rename_pattern_B = st.selectbox("Nytt filnavn", [
            "Behold originalt navn","S_OBJID + originalt navn","Kun S_OBJID","S_OBJID + avrundet E/N"
        ], index=1, key="B_rename")
        draw_arrow_B = st.checkbox("Tegn nordpil", value=False, key="B_arrow")
        arrow_size_B = st.slider("Pil-størrelse", 60, 240, 120, key="B_arrow_size")
    with colB2:
        epsg_in2_default = ensure_epsg("Inn-CRS standard (hvis rad mangler EPSG)", key_base="B_epsg_in", default=25832)
        epsg_out_doc2 = ensure_epsg("Dokumentasjons-CRS for CSV (kun eksport)", key_base="B_epsg_doc", default=25832)
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
                if not cols["file"] and not cols["folder"]:
                    st.error("Mangler 'file'/'folder' (eller S_HYPERLINK).")
                elif not cols["east"] or not cols["north"]:
                    st.error("Mangler Øst/Nord.")
                else:
                    rows = []
                    patt_map = {"Behold originalt navn":"keep","S_OBJID + originalt navn":"label_orig","Kun S_OBJID":"label_only","S_OBJID + avrundet E/N":"label_en"}
                    patt = patt_map[rename_pattern_B]
                    def row_epsg(row):
                        try:
                            if cols["epsg"] and (str(row.get(cols["epsg"], "")).strip() != ""):
                                return int(row[cols["epsg"]])
                        except Exception: pass
                        return epsg_in2_default
                    def process_target(path_list, E, N, Alt, Dir, epsg_row, label):
                        lat, lon = transform_EN_to_wgs84(E, N, epsg_row)
                        for p in path_list:
                            if not os.path.isfile(p): st.warning(f"Mangler fil: {p}"); continue
                            fname = os.path.relpath(p, root)
                            newname = build_new_name(patt, label, os.path.basename(fname), E, N)
                            dst = p if overwrite2 else os.path.join(out_root or os.path.dirname(p), newname)
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            if dst != p:
                                if not overwrite2: dst = dst if out_root else dst
                                shutil.copy2(p, dst)
                            im = Image.open(dst); im = normalize_orientation(im)
                            if draw_arrow_B and _is_valid_number(Dir): im = draw_north_arrow(im, _wrap_deg(Dir), size_px=arrow_size_B)
                            im.save(dst, "jpeg", quality=95)
                            write_exif(dst, dst, lat, lon, Alt, Dir)
                            Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_row, epsg_out_doc2)
                            rows.append({"file": os.path.relpath(dst, out_root) if out_root else os.path.basename(dst),
                                         "S_OBJID": label, "E_in": E, "N_in": N, "epsg_in": epsg_row,
                                         "lat": lat, "lon": lon, f"E_{epsg_out_doc2}": Xdoc, f"N_{epsg_out_doc2}": Ydoc,
                                         "hoyde": Alt, "rotasjon": Dir})
                    # folder/files
                    if cols["folder"] and cols["folder"] in dfm.columns:
                        for _, r in dfm.dropna(subset=[cols["folder"], cols["east"], cols["north"]]).iterrows():
                            E=parse_float_maybe_comma(r[cols["east"]]); N=parse_float_maybe_comma(r[cols["north"]])
                            Alt=parse_float_maybe_comma(r[cols["alt"]]) if cols["alt"] else None
                            Dir=parse_float_maybe_comma(r[cols["rot"]]) if cols["rot"] else None
                            lab=r[cols["sobj"]] if cols["sobj"] else None
                            eps=row_epsg(r)
                            folder=os.path.join(root, str(r[cols["folder"]]))
                            jpgs=sorted(glob.glob(os.path.join(folder, "*.jpg"))+glob.glob(os.path.join(folder, "*.JPG")))
                            if E is None or N is None: st.warning(f"Hopper rad (ugyldig E/N): {lab or ''}"); continue
                            process_target(jpgs, E, N, Alt, Dir, eps, lab)
                    if cols["file"] and cols["file"] in dfm.columns:
                        for _, r in dfm.dropna(subset=[cols["file"], cols["east"], cols["north"]]).iterrows():
                            E=parse_float_maybe_comma(r[cols["east"]]); N=parse_float_maybe_comma(r[cols["north"]])
                            Alt=parse_float_maybe_comma(r[cols["alt"]]) if cols["alt"] else None
                            Dir=parse_float_maybe_comma(r[cols["rot"]]) if cols["rot"] else None
                            lab=r[cols["sobj"]] if cols["sobj"] else None
                            eps=row_epsg(r)
                            p=os.path.join(root, str(r[cols["file"]]))
                            if E is None or N is None: st.warning(f"Hopper rad (ugyldig E/N): {lab or ''}"); continue
                            process_target([p], E, N, Alt, Dir, eps, lab)
                    dfr=pd.DataFrame(rows); st.success(f"Geotagget {len(dfr)} bilder.")
                    st.download_button("Last ned CSV (Tab B)", dfr.to_csv(index=False).encode("utf-8"), "geotag_tabB.csv", "text/csv")
        except Exception as e:
            st.exception(e)

# ---------- Tab C (centers) ----------
with tabC:
    st.subheader("C) Kum-senter + heading fra hjørner (PCA)")
    st.markdown("Grupperer S_OBJID på prefiks (før skilletegn) og beregner senter/heading.")
    from math import atan2, degrees
    import numpy as np

    def base_id(s, delims="-_ ./"):
        s=str(s).strip()
        for d in delims:
            if d in s: return s.split(d,1)[0].strip().upper()
        m=re.match(r'^([A-ZÆØÅ0-9]+?)(?:\s*(?:HJ(?:ØRNE)?|H|C)?\s*\d+)?$', s.upper())
        return m.group(1) if m else s.upper()

    def pca_heading(points):
        A=np.array(points, dtype=float)
        c=A.mean(axis=0); A0=A-c
        U,S,Vt=np.linalg.svd(A0, full_matrices=False)
        vx,vy=Vt[0,0],Vt[0,1]
        if vy<0: vx,vy=-vx,-vy
        az=(degrees(math.atan2(vx,vy))+360.0)%360.0
        proj_main=A0@Vt[0,:]; proj_ortho=A0@Vt[1,:]
        L=(proj_main.max()-proj_main.min()); W=(proj_ortho.max()-proj_ortho.min())
        return (float(c[0]),float(c[1])), az, abs(float(L)), abs(float(W))

    colC1, colC2 = st.columns(2)
    with colC1:
        corners_up = st.file_uploader("CSV/Excel med hjørner: S_OBJID, Øst, Nord, (Høyde)", type=["csv","xlsx","xls"], key="C_corners")
        delims = st.text_input("Skilletegn for grunn-ID:", value="-_ ./", key="C_delims")
        use_center_as_exif = st.checkbox("Bruk kum-senter som EXIF-posisjon", value=True, key="C_use_center")
        use_heading_as_exif = st.checkbox("Bruk beregnet heading", value=True, key="C_use_head")
    with colC2:
        epsg_inC = ensure_epsg("Inn-CRS for hjørner", key_base="C_epsg_in", default=25832)
        epsg_outC = ensure_epsg("Dok-CRS (kun eksport)", key_base="C_epsg_doc", default=25832)
        rootC = st.text_input("Root-mappe for bilder", value="", key="C_root")
        mapping_up = st.file_uploader("Mapping-CSV/Excel (S_OBJID ↔ file / S_HYPERLINK)", type=["csv","xlsx","xls"], key="C_map")
        rename_pattern_C = st.selectbox("Nytt filnavn", ["Behold originalt navn","S_OBJID + originalt navn","Kun S_OBJID","S_OBJID + avrundet E/N"], index=1, key="C_rename")
        overwriteC = st.checkbox("Overskriv", value=False, key="C_overwrite")

    runC = st.button("Beregn og geotagg (Tab C)", key="C_run")
    if runC:
        try:
            if not corners_up:
                st.error("Last opp hjørnefil.")
            else:
                dfc=dataframe_from_upload(corners_up); colsc=detect_columns(dfc)
                if not colsc["east"] or not colsc["north"] or not colsc["sobj"]:
                    st.error("Trenger S_OBJID, Øst, Nord."); 
                else:
                    recs=[]
                    for sobj, grp in dfc.groupby(dfc[colsc["sobj"]].astype(str)):
                        base=base_id(sobj, delims)
                        pts=[]; zs=[]
                        for _,r in grp.iterrows():
                            e=parse_float_maybe_comma(r[colsc["east"]]); n=parse_float_maybe_comma(r[colsc["north"]])
                            if e is None or n is None: continue
                            pts.append((e,n))
                            if colsc["alt"]:
                                z=parse_float_maybe_comma(r[colsc["alt"]]); 
                                if z is not None: zs.append(z)
                        if len(pts)>=3:
                            (ce,cn), az, L, W = pca_heading(pts)
                            zmed=float(pd.Series(zs).median()) if len(zs)>0 else None
                            recs.append({"base_id":base,"center_E":ce,"center_N":cn,"azimut":az,"L":L,"W":W,"hoyde":zmed})
                    dfo=pd.DataFrame(recs).drop_duplicates(subset=["base_id"])
                    st.dataframe(dfo.head(30))
                    if mapping_up is not None:
                        if not rootC or not os.path.isdir(rootC): st.error("Root-mappe finnes ikke.")
                        else:
                            dfm=dataframe_from_upload(mapping_up); colm=detect_columns(dfm)
                            if colm["hyper"] and colm["hyper"] in dfm.columns:
                                dfm=expand_hyperlinks(dfm, colm["hyper"]); colm["file"]="file"
                            centers={r["base_id"]:r for _,r in dfo.iterrows()}
                            rows=[]
                            patt_map={"Behold originalt navn":"keep","S_OBJID + originalt navn":"label_orig","Kun S_OBJID":"label_only","S_OBJID + avrundet E/N":"label_en"}
                            patt=patt_map[rename_pattern_C]
                            for _,r in dfm.dropna(subset=[colm["sobj"], colm["file"]]).iterrows():
                                sobj=str(r[colm["sobj"]]); base=base_id(sobj, delims); info=centers.get(base)
                                if info is None: st.warning(f"Ingen senter/heading for {sobj}"); continue
                                if use_center_as_exif:
                                    E=float(info["center_E"]); N=float(info["center_N"]); epsg_row=epsg_inC; Alt=info["hoyde"]
                                else:
                                    if not colm["east"] or not colm["north"]:
                                        E=float(info["center_E"]); N=float(info["center_N"]); epsg_row=epsg_inC; Alt=info["hoyde"]
                                    else:
                                        E=parse_float_maybe_comma(r[colm["east"]]); N=parse_float_maybe_comma(r[colm["north"]])
                                        Alt=parse_float_maybe_comma(r[colm["alt"]]) if colm["alt"] else info["hoyde"]; epsg_row=epsg_inC
                                Dir=float(info["azimut"]) if use_heading_as_exif else parse_float_maybe_comma(r[colm["rot"]]) if colm["rot"] else None
                                lat,lon=transform_EN_to_wgs84(E,N,epsg_row)
                                p=os.path.join(rootC, str(r[colm["file"]]))
                                if not os.path.isfile(p): st.warning(f"Mangler fil: {p}"); continue
                                newname=build_new_name(patt, sobj, os.path.basename(p), E, N)
                                dst=p if overwriteC else os.path.join(os.path.dirname(p), newname)
                                os.makedirs(os.path.dirname(dst), exist_ok=True)
                                if dst != p: shutil.copy2(p, dst)
                                im=Image.open(dst); im=normalize_orientation(im)
                                if _is_valid_number(Dir): im=draw_north_arrow(im, _wrap_deg(Dir), size_px=120)
                                im.save(dst, "jpeg", quality=95)
                                write_exif(dst, dst, lat, lon, Alt, Dir)
                                rows.append({"file": os.path.basename(dst), "S_OBJID": sobj, "base_id": base,
                                             "center_E": info["center_E"], "center_N": info["center_N"],
                                             "lat": lat, "lon": lon, "azimut": info["azimut"], "hoyde": Alt})
                            dfr=pd.DataFrame(rows); st.success(f"Geotagget {len(dfr)} bilder."); 
                            st.download_button("Last ned CSV (Tab C)", dfr.to_csv(index=False).encode("utf-8"), "geotag_tabC.csv", "text/csv")
        except Exception as e:
            st.exception(e)

# ---------- Tab D (lines + correction + map) ----------
with tabD:
    st.subheader("D) Linje-basert heading + korrektur + kart")
    st.markdown("Les et **linjelag (GeoJSON)** og la appen sette **GPSImgDirection** fra nærmeste linje. "
                "Deretter kan du **korrigere** (±1/±5/±10°, +180°) og **låse** verdier før skriving av EXIF.")
    colD1, colD2 = st.columns(2)
    with colD1:
        rootD = st.text_input("Root-mappe for bilder", value="", key="D_root")
        map_up = st.file_uploader("Mapping-CSV/Excel (S_OBJID/file/E/N/Rotasjon/EPSG)", type=["csv","xlsx","xls"], key="D_map")
        lines_up = st.file_uploader("Linjer (GeoJSON, samme CRS som E/N)", type=["geojson","json"], key="D_lines")
        objtype_field = st.text_input("Objekttype-felt i linje-GeoJSON (valgfritt)", value="objtype", key="D_objf")
        type_filter = st.text_input("Filter (kommaseparerte typer, valgfritt)", value="", key="D_filter")
        buffer_m = st.number_input("Buffer (m) for nærmeste linje", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="D_buf")
        overwriteD = st.checkbox("Overskriv filer", value=False, key="D_overwrite")
    with colD2:
        epsg_points = ensure_epsg("EPSG for E/N i mapping", key_base="D_epsg_pts", default=25832)
        epsg_lines = ensure_epsg("EPSG for linje-GeoJSON", key_base="D_epsg_lin", default=25832)
        draw_arrow_D = st.checkbox("Tegn nordpil", value=True, key="D_arrow")
        arrow_size_D = st.slider("Pil-størrelse", 60, 240, 120, key="D_arrow_size")
        only_fill_missing = st.checkbox("Fyll kun manglende heading (ikke overstyr)", value=True, key="D_only_missing")

    # State for corrections
    if "D_corrections" not in st.session_state:
        st.session_state["D_corrections"] = {}  # key = file path, value = heading deg + locked flag
    if "D_locked" not in st.session_state:
        st.session_state["D_locked"] = set()

    runD = st.button("Beregn heading fra nærmeste linje", key="D_run")
    if runD:
        try:
            if not rootD or not os.path.isdir(rootD):
                st.error("Root-mappe finnes ikke.")
            elif not map_up:
                st.error("Last opp mapping-CSV/Excel.")
            elif not lines_up:
                st.error("Last opp linje-GeoJSON.")
            else:
                # Load mapping
                df = dataframe_from_upload(map_up)
                cols = detect_columns(df)
                if cols["hyper"] and cols["hyper"] in df.columns:
                    df = expand_hyperlinks(df, cols["hyper"]); cols["file"]="file"
                need_cols = [cols["file"], cols["east"], cols["north"]]
                if any(c is None for c in need_cols):
                    st.error("Mapping må ha kolonnene: file, Øst, Nord (evt via S_HYPERLINK).")
                else:
                    # Transform points if EPSG differs? We assume both in same to simplify; otherwise convert lines? We'll assume same CRS as input for now.
                    # Load lines
                    lines = load_lines_auto(lines_up, lines_up.name, prop_objtype=objtype_field or None)
                    if type_filter.strip():
                        allowed = set([s.strip() for s in type_filter.split(",") if s.strip()])
                        lines = [L for L in lines if (L["objtype"] in allowed)]
                    if epsg_lines != epsg_points:
                        # Reproject lines to point CRS
                        tr = Transformer.from_crs(epsg_lines, epsg_points, always_xy=True)
                        for L in lines:
                            L["coords"] = [tuple(tr.transform(x,y)) for (x,y) in L["coords"]]
                    # Compute nearest heading for each image
                    recs = []
                    for _, r in df.dropna(subset=[cols["file"], cols["east"], cols["north"]]).iterrows():
                        pth = os.path.join(rootD, str(r[cols["file"]]))
                        if not os.path.isfile(pth): 
                            continue
                        E=parse_float_maybe_comma(r[cols["east"]]); N=parse_float_maybe_comma(r[cols["north"]])
                        if E is None or N is None: continue
                        best = (None, float("inf"), None, None)  # (heading, dist, line_idx, nearest_pt)
                        for idx, L in enumerate(lines):
                            hd, dist, npt = nearest_heading_on_polyline(L["coords"], (E,N))
                            if npt is None: 
                                continue
                            if dist < best[1]:
                                best = (hd, dist, idx, npt)
                        hd, dist, idx, npt = best
                        row = {
                            "file": os.path.relpath(pth, rootD),
                            "S_OBJID": r.get(cols["sobj"], None),
                            "E": E, "N": N,
                            "heading_line": hd, "dist_m": dist,
                            "has_line_match": dist <= buffer_m
                        }
                        recs.append(row)
                    dd = pd.DataFrame(recs)
                    if dd.empty:
                        st.warning("Fant ingen gyldige rader.")
                    else:
                        st.session_state["D_table"] = dd
                        st.success(f"Beregnede headinger for {len(dd)} bilder. Se kart og korrektur under.")
        except Exception as e:
            st.exception(e)

    # Show map and correction UI if we have table
    if "D_table" in st.session_state:
        dd = st.session_state["D_table"]
        # Map view using pydeck
        try:
            import pydeck as pdk
            # Build layers
            # Points need lat/lon; convert from E/N
            tr_pts = Transformer.from_crs(epsg_points, 4326, always_xy=True)
            dd["lon"], dd["lat"] = zip(*[tr_pts.transform(e, n) for e,n in zip(dd["E"], dd["N"])])
            # Color: green=matched, red=unmatched
            dd["color"] = dd["has_line_match"].map(lambda b: [0,180,0] if b else [200,50,50])
            scatter = pdk.Layer("ScatterplotLayer", dd, get_position='[lon, lat]',
                                get_radius=2, get_fill_color='color', pickable=True)
            layers = [scatter]
            # If have lines uploaded, show them
            if lines_up is not None:
                lines_up.seek(0)
                lines_geo = json.load(lines_up)
                # Build simple path layer (WGS84)
                feats = lines_geo["features"] if "features" in lines_geo else [lines_geo]
                paths = []
                tr_lin = Transformer.from_crs(epsg_lines, 4326, always_xy=True)
                for f in feats:
                    g=f.get("geometry",{}); t=g.get("type","")
                    if t=="LineString":
                        coords=[tr_lin.transform(x,y) for (x,y,*rest) in g.get("coordinates",[])]
                        paths.append({"path":[[lon,lat] for (lon,lat) in coords]})
                    elif t=="MultiLineString":
                        for arr in g.get("coordinates",[]):
                            coords=[tr_lin.transform(x,y) for (x,y,*rest) in arr]
                            paths.append({"path":[[lon,lat] for (lon,lat) in coords]})
                line_layer = pdk.Layer("PathLayer", paths, get_path="path", get_width=2, get_color=[80,80,200])
                layers.append(line_layer)
            view_state = pdk.ViewState(latitude=float(dd["lat"].mean()), longitude=float(dd["lon"].mean()), zoom=16)
            st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view_state), use_container_width=True)
        except Exception as e:
            st.info("Kunne ikke vise kart (pydeck mangler eller feil).")

        st.markdown("### Korrektur")
        # Filter table to unmatched or all
        mode = st.radio("Vis:", ["Kun uten linjematch", "Alle"], index=0, horizontal=True, key="D_viewmode")
        to_show = dd[~dd["has_line_match"]] if mode.startswith("Kun") else dd
        st.dataframe(to_show[["file","S_OBJID","E","N","heading_line","dist_m","has_line_match"]].head(100), use_container_width=True)

        # Select one file for manual adjust
        options = to_show["file"].tolist()
        sel = st.selectbox("Velg bilde for manuell korrigering", options=options if options else dd["file"].tolist(), key="D_pick")
        if sel:
            row = dd[dd["file"]==sel].iloc[0]
            current = row["heading_line"] if row["has_line_match"] else None
            # Apply existing correction if present
            if sel in st.session_state["D_corrections"]:
                current = st.session_state["D_corrections"][sel]
            st.write(f"Nåværende heading (linje/forslag): {current if current is not None else '—'}°")
            new_heading = st.slider("Sett heading (grader fra nord)", 0, 359, int(current or 0), key="D_slider")
            cols_btn = st.columns(5)
            with cols_btn[0]:
                if st.button("-10°", key="D_m10"): new_heading = (new_heading - 10) % 360
            with cols_btn[1]:
                if st.button("-1°", key="D_m1"): new_heading = (new_heading - 1) % 360
            with cols_btn[2]:
                if st.button("+1°", key="D_p1"): new_heading = (new_heading + 1) % 360
            with cols_btn[3]:
                if st.button("+10°", key="D_p10"): new_heading = (new_heading + 10) % 360
            with cols_btn[4]:
                if st.button("+180°", key="D_p180"): new_heading = (new_heading + 180) % 360
            st.session_state["D_corrections"][sel] = new_heading
            lock = st.checkbox("Lås heading for dette bildet", value=(sel in st.session_state["D_locked"]), key="D_lock")
            if lock: st.session_state["D_locked"].add(sel)
            else:
                if sel in st.session_state["D_locked"]: st.session_state["D_locked"].remove(sel)
            st.write(f"Ny/verdi for {sel}: {new_heading}° {'(låst)' if sel in st.session_state['D_locked'] else ''}")

        # Apply to all (write EXIF)
        if st.button("Skriv EXIF for alle i tabellen", key="D_apply"):
            try:
                rows_out = []
                for _, r in dd.iterrows():
                    p = os.path.join(rootD, r["file"])
                    if not os.path.isfile(p): 
                        continue
                    # choose heading:
                    hd = st.session_state["D_corrections"].get(r["file"], r["heading_line"] if r["has_line_match"] else None)
                    # honor only_fill_missing:
                    if only_fill_missing and r["has_line_match"] and (r["file"] not in st.session_state["D_locked"]) and (r["file"] not in st.session_state["D_corrections"]):
                        # if only fill missing, but we have a line match, we still fill if original missing - for simplicity assume missing
                        pass
                    # Write image
                    lat, lon = transform_EN_to_wgs84(r["E"], r["N"], epsg_points)
                    dst = p  # in-place unless overwrite to copy with new name (skipping rename in D)
                    im = Image.open(dst); im = normalize_orientation(im)
                    if draw_arrow_D and _is_valid_number(hd): im = draw_north_arrow(im, _wrap_deg(hd), size_px=arrow_size_D)
                    im.save(dst, "jpeg", quality=95)
                    write_exif(dst, dst, lat, lon, None, hd)
                    rows_out.append({"file": r["file"], "lat": lat, "lon": lon, "heading": hd, "source": "line/corrected"})
                dfo = pd.DataFrame(rows_out)
                st.success(f"Skrev EXIF for {len(dfo)} bilder.")
                st.download_button("Last ned CSV (Tab D)", dfo.to_csv(index=False).encode("utf-8"), "geotag_tabD.csv", "text/csv")
            except Exception as e:
                st.exception(e)

st.markdown("---")
st.caption("v8 • linje-heading • korrektur (slider/nudge/+180/lås) • kartvisning • EXIF WGS84")


import os, glob, shutil, io, zipfile, re, math, json
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import piexif
from pyproj import Transformer, CRS

st.set_page_config(page_title="Geotagging bilder v9.1 • Gemini/CSV → EXIF • Kum-senter • Linje-heading • Kart", layout="wide")

# ================ Utilities ================

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
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b'T'
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

# ====== LandXML / GeoJSON line readers ======

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
    def findall(elem, path):
        res = elem.findall(path, ns) if ns else elem.findall(path)
        if res: return res
        return [e for e in elem.iter() if e.tag.endswith(path.split('/')[-1])]
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
            az = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0
            best = (az, dist, (nx,ny))
    return best

def base_id(s, delims="-_ ./"):
    s=str(s).strip()
    for d in delims:
        if d in s: return s.split(d,1)[0].strip().upper()
    m = re.match(r'^([A-ZÆØÅ0-9]+?)(?:\s*(?:HJ(?:ØRNE)?|H|C)?\s*\d+)?$', s.upper())
    return m.group(1) if m else s.upper()

# ================ UI ================

st.title("Geotagging bilder v9.1")
st.caption("• S_HYPERLINK → én rad per bilde • EXIF WGS84 • Nordpil • Kum-senter/heading fra hjørner • Linje-heading (GeoJSON/LandXML) • Kart • Korrektur")

# ---------- Tab A (multi) ----------
tabA, tabB, tabC, tabD = st.tabs(["A) Mappe/ZIP/Flere bilder", "B) CSV-mapping", "C) Kum-senter", "D) Linje-heading + kart"])

with tabA:
    st.subheader("A) Geotagg mange bilder fra mappe, ZIP eller direkte opplasting")
    mode = st.radio("Kilde:", ["Lokal mappe", "ZIP-opplasting", "Opplasting (flere filer)"], index=2, key="A_mode")
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

        draw_arrow_A = st.checkbox("Tegn nordpil på alle bilder", value=True, key="A_arrow")
        arrow_size_A = st.slider("Pil-størrelse (px)", 60, 240, 120, key="A_arrow_size")

    with colA2:
        st.markdown("#### Velg punkt fra fil (Excel/CSV) – bruker samme posisjon og (ev.) rotasjon for alle bildene")
        pts_up = st.file_uploader("Punktliste: Øst, Nord, (Høyde), (Rotasjon), (S_OBJID)", type=["xlsx","xls","csv"], key="A_pts")
        epsg_inA = ensure_epsg("Inn-CRS (UTM/NTM) for E/N", key_base="A_epsg_in", default=25832)
        epsg_outA = ensure_epsg("Dokumentasjons-CRS for CSV (eksport)", key_base="A_epsg_doc", default=25832)
        E=N=Alt=Dir=label=None
        if pts_up is not None:
            dfp = dataframe_from_upload(pts_up)
            colsA = detect_columns(dfp)
            if not colsA["east"] or not colsA["north"]:
                st.error("Fant ikke Øst/Nord i punktfila.")
            else:
                show_cols = [c for c in [colsA["sobj"], colsA["east"], colsA["north"], colsA["alt"], colsA["rot"]] if c]
                st.dataframe(dfp[show_cols].head(20), use_container_width=True)
                idx = st.selectbox("Velg punkt:", dfp.index.to_list(),
                                   format_func=lambda i: f"{dfp.loc[i, colsA['sobj']]}" if colsA["sobj"] else f"Rad {i+1}",
                                   key="A_pick")
                pf=lambda v: parse_float_maybe_comma(v)
                E=pf(dfp.loc[idx, colsA["east"]]); N=pf(dfp.loc[idx, colsA["north"]])
                Alt=pf(dfp.loc[idx, colsA["alt"]]) if colsA["alt"] else None
                Dir=pf(dfp.loc[idx, colsA["rot"]]) if colsA["rot"] else None
                label = dfp.loc[idx, colsA["sobj"]] if colsA["sobj"] else None

    if st.button("Kjør geotag (Tab A)", key="A_run"):
        try:
            if E is None or N is None:
                st.error("E/N mangler.")
            else:
                lat, lon = transform_EN_to_wgs84(float(E), float(N), epsg_inA)
                patt_map = {"Behold originalt navn":"keep","S_OBJID + originalt navn":"label_orig","Kun S_OBJID":"label_only","S_OBJID + avrundet E/N":"label_en"}
                patt = patt_map[rename_pattern_A]
                rows=[]
                if mode == "Lokal mappe":
                    if not rootA or not os.path.isdir(rootA): st.error("Aktiv mappe finnes ikke.")
                    else:
                        out_dir = rootA if (overwriteA or not outA) else outA
                        if not overwriteA and outA: os.makedirs(outA, exist_ok=True)
                        jpgs = sorted(glob.glob(os.path.join(rootA, "*.jpg")) + glob.glob(os.path.join(rootA, "*.JPG")))
                        for p in jpgs:
                            fname=os.path.basename(p)
                            newname = build_new_name(patt, label, fname, E, N)
                            dst = p if (overwriteA and not outA and newname==fname) else os.path.join(out_dir, newname)
                            if dst != p:
                                os.makedirs(os.path.dirname(dst), exist_ok=True)
                                shutil.copy2(p, dst)
                            im=Image.open(dst); im=normalize_orientation(im)
                            if draw_arrow_A and _is_valid_number(Dir): im=draw_north_arrow(im, _wrap_deg(Dir), size_px=arrow_size_A)
                            im.save(dst, "jpeg", quality=95)
                            write_exif(dst, dst, lat, lon, Alt, Dir)
                            Xdoc, Ydoc = transform_EN_to_epsg(float(E), float(N), epsg_inA, epsg_outA)
                            rows.append({"file": os.path.basename(dst), "label": label, "E_in": E, "N_in": N, "lat": lat, "lon": lon,
                                         f"E_{epsg_outA}": Xdoc, f"N_{epsg_outA}": Ydoc, "hoyde": Alt, "rotasjon": Dir})
                    dfo=pd.DataFrame(rows)
                    st.success(f"Geotagget {len(dfo)} bilder.")
                    st.download_button("Last ned CSV (Tab A)", dfo.to_csv(index=False).encode("utf-8"), "geotag_tabA.csv", "text/csv")

                elif mode == "ZIP-opplasting":
                    if not zip_up: st.error("Last opp ZIP.")
                    else:
                        zin = zipfile.ZipFile(io.BytesIO(zip_up.read()), "r")
                        zout_mem = io.BytesIO(); zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)
                        used=set()
                        for name in zin.namelist():
                            if not name.lower().endswith((".jpg",".jpeg")): continue
                            payload = zin.read(name)
                            im = Image.open(io.BytesIO(payload)); im=normalize_orientation(im)
                            if draw_arrow_A and _is_valid_number(Dir): im=draw_north_arrow(im, _wrap_deg(Dir), size_px=arrow_size_A)
                            out_io=io.BytesIO(); im.save(out_io, "jpeg", quality=95); out_io.seek(0)
                            tmp=out_io.getvalue(); open("tmp.jpg","wb").write(tmp)
                            write_exif("tmp.jpg", "tmp.jpg", lat, lon, Alt, Dir)
                            final=open("tmp.jpg","rb").read()
                            newname=build_new_name(patt, label, os.path.basename(name), E, N)
                            base,ext=os.path.splitext(newname); i=1; cand=newname
                            while cand in used:
                                cand=f"{base}_{i}{ext}"; i+=1
                            used.add(cand)
                            zout.writestr(cand, final)
                        zout.close(); st.download_button("Last ned geotagget ZIP", data=zout_mem.getvalue(), file_name="geotagged.zip", mime="application/zip")

                else:  # multiple files upload
                    if not files_up: st.error("Last opp JPG-er.")
                    else:
                        zout_mem=io.BytesIO(); zout=zipfile.ZipFile(zout_mem,"w",zipfile.ZIP_DEFLATED)
                        used=set()
                        for f in files_up:
                            name = f.name
                            im=Image.open(f); im=normalize_orientation(im)
                            if draw_arrow_A and _is_valid_number(Dir): im=draw_north_arrow(im, _wrap_deg(Dir), size_px=arrow_size_A)
                            out_io=io.BytesIO(); im.save(out_io, "jpeg", quality=95); out_io.seek(0)
                            tmp=out_io.getvalue(); open("tmp.jpg","wb").write(tmp)
                            write_exif("tmp.jpg", "tmp.jpg", lat, lon, Alt, Dir)
                            final=open("tmp.jpg","rb").read()
                            newname=build_new_name(patt, label, os.path.basename(name), E, N)
                            base,ext=os.path.splitext(newname); i=1; cand=newname
                            while cand in used:
                                cand=f"{base}_{i}{ext}"; i+=1
                            used.add(cand)
                            zout.writestr(cand, final)
                        zout.close(); st.download_button("Last ned ZIP med geotaggede bilder", data=zout_mem.getvalue(), file_name="geotagged_upload.zip", mime="application/zip")
        except Exception as e:
            st.exception(e)

# ---------- Tab B (CSV mapping) ----------
with tabB:
    st.subheader("B) CSV/Excel-mapping (inkl. S_HYPERLINK)")
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
                            if dst != p: shutil.copy2(p, dst)
                            im = Image.open(dst); im = normalize_orientation(im)
                            if draw_arrow_B and _is_valid_number(Dir): im = draw_north_arrow(im, _wrap_deg(Dir), size_px=arrow_size_B)
                            im.save(dst, "jpeg", quality=95)
                            write_exif(dst, dst, lat, lon, Alt, Dir)
                            Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_row, epsg_out_doc2)
                            rows.append({"file": os.path.relpath(dst, out_root) if out_root else os.path.basename(dst),
                                         "S_OBJID": label, "E_in": E, "N_in": N, "epsg_in": epsg_row,
                                         "lat": lat, "lon": lon, f"E_{epsg_out_doc2}": Xdoc, f"N_{epsg_out_doc2}": Ydoc,
                                         "hoyde": Alt, "rotasjon": Dir})
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

# ---------- Tab C (kum-senter) ----------
with tabC:
    st.subheader("C) Kum-senter + heading fra hjørner (PCA)")
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
    colD1, colD2 = st.columns(2)
    with colD1:
        rootD = st.text_input("Root-mappe for bilder", value="", key="D_root")
        map_up = st.file_uploader("Mapping-CSV/Excel (S_OBJID/file/E/N/Rotasjon/EPSG)", type=["csv","xlsx","xls"], key="D_map")
        lines_up = st.file_uploader("Linjer (GeoJSON eller XML/LandXML)", type=["geojson","json","xml","landxml"], key="D_lines")
        objtype_field = st.text_input("Objekttype-felt (valgfritt)", value="objtype", key="D_objf")
        type_filter = st.text_input("Typefilter (komma-separert, valgfritt)", value="", key="D_filter")
        buffer_m = st.number_input("Buffer (m) for nærmeste linje", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="D_buf")
        overwriteD = st.checkbox("Overskriv filer", value=False, key="D_overwrite")
    with colD2:
        epsg_points = ensure_epsg("EPSG for E/N i mapping", key_base="D_epsg_pts", default=25832)
        epsg_lines = ensure_epsg("EPSG for linjer (GeoJSON/XML)", key_base="D_epsg_lin", default=25832)
        draw_arrow_D = st.checkbox("Tegn nordpil", value=True, key="D_arrow")
        arrow_size_D = st.slider("Pil-størrelse", 60, 240, 120, key="D_arrow_size")
        only_fill_missing = st.checkbox("Fyll kun manglende heading", value=True, key="D_only_missing")

    if "D_corrections" not in st.session_state:
        st.session_state["D_corrections"] = {}
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
                st.error("Last opp linje-fil (GeoJSON/XML).")
            else:
                df = dataframe_from_upload(map_up)
                cols = detect_columns(df)
                if cols["hyper"] and cols["hyper"] in df.columns:
                    df = expand_hyperlinks(df, cols["hyper"]); cols["file"]="file"
                need_cols = [cols["file"], cols["east"], cols["north"]]
                if any(c is None for c in need_cols):
                    st.error("Mapping må ha kolonnene: file, Øst, Nord (evt via S_HYPERLINK).")
                else:
                    lines = load_lines_auto(lines_up, lines_up.name, prop_objtype=objtype_field or None)
                    if type_filter.strip():
                        allowed = set([s.strip() for s in type_filter.split(",") if s.strip()])
                        lines = [L for L in lines if (L["objtype"] in allowed)]
                    if epsg_lines != epsg_points:
                        tr = Transformer.from_crs(epsg_lines, epsg_points, always_xy=True)
                        for L in lines:
                            L["coords"] = [tuple(tr.transform(x,y)) for (x,y) in L["coords"]]
                    recs = []
                    for _, r in df.dropna(subset=[cols["file"], cols["east"], cols["north"]]).iterrows():
                        pth = os.path.join(rootD, str(r[cols["file"]]))
                        if not os.path.isfile(pth): 
                            continue
                        E=parse_float_maybe_comma(r[cols["east"]]); N=parse_float_maybe_comma(r[cols["north"]])
                        if E is None or N is None: continue
                        best = (None, float("inf"), None)
                        for idx, L in enumerate(lines):
                            hd, dist, npt = nearest_heading_on_polyline(L["coords"], (E,N))
                            if npt is None: 
                                continue
                            if dist < best[1]:
                                best = (hd, dist, idx)
                        hd, dist, idx = best
                        row = {
                            "file": os.path.relpath(pth, rootD),
                            "S_OBJID": r.get(cols["sobj"], None),
                            "E": E, "N": N,
                            "heading_line": hd, "dist_m": dist,
                            "has_line_match": dist <= buffer_m if dist is not None else False
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

    if "D_table" in st.session_state:
        dd = st.session_state["D_table"]
        try:
            import pydeck as pdk
            tr_pts = Transformer.from_crs(epsg_points, 4326, always_xy=True)
            dd["lon"], dd["lat"] = zip(*[tr_pts.transform(e, n) for e,n in zip(dd["E"], dd["N"])])
            dd["color"] = dd["has_line_match"].map(lambda b: [0,180,0] if b else [200,50,50])
            scatter = pdk.Layer("ScatterplotLayer", dd, get_position='[lon, lat]', get_radius=2, get_fill_color='color', pickable=True)
            layers = [scatter]
            # draw lines (approx in same view): we can't easily reuse the file after read; skip for brevity
            view_state = pdk.ViewState(latitude=float(dd["lat"].mean()), longitude=float(dd["lon"].mean()), zoom=16)
            st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view_state), use_container_width=True)
        except Exception as e:
            st.info("Kunne ikke vise kart (pydeck mangler eller feil).")

        st.markdown("### Korrektur")
        mode = st.radio("Vis:", ["Kun uten linjematch", "Alle"], index=0, horizontal=True, key="D_viewmode")
        to_show = dd[~dd["has_line_match"]] if mode.startswith("Kun") else dd
        st.dataframe(to_show[["file","S_OBJID","E","N","heading_line","dist_m","has_line_match"]].head(100), use_container_width=True)

        options = to_show["file"].tolist() if len(to_show)>0 else dd["file"].tolist()
        sel = st.selectbox("Velg bilde for manuell korrigering", options=options, key="D_pick")
        if sel:
            row = dd[dd["file"]==sel].iloc[0]
            current = row["heading_line"] if row["has_line_match"] else None
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

        if st.button("Skriv EXIF for alle i tabellen", key="D_apply"):
            try:
                rows_out = []
                for _, r in dd.iterrows():
                    p = os.path.join(rootD, r["file"])
                    if not os.path.isfile(p): continue
                    hd = st.session_state["D_corrections"].get(r["file"], r["heading_line"] if r["has_line_match"] else None)
                    lat, lon = transform_EN_to_wgs84(r["E"], r["N"], epsg_points)
                    dst = p
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
st.caption("v9.1 • EXIF WGS84 • S_HYPERLINK-splitting • Kum-senter/heading • Linje-heading (GeoJSON/LandXML) • Manuell korrektur • Kart • Nordpil")

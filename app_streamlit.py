
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

st.set_page_config(page_title="Geotagging bilder v11.3", layout="wide")
st.title("Geotagging bilder v11.3")
st.caption("v11.3 • Hjørnepunkter: mye mindre mulig, valg av etikett (idx/punktnavn), linjebredde-justering • + alt fra v11.2")

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
    pnt_by_name = {}
    for p in root.iter():
        if p.tag.endswith("Pnt") and p.get("name"):
            nums = _parse_numbers_list(p.text or "")
            if len(nums)>=2: pnt_by_name[p.get("name")] = (nums[0], nums[1])
    lines = []
    for tag in ["PntList2D","PntList3D"]:
        for pl in root.iter():
            if pl.tag.endswith(tag):
                nums = _parse_numbers_list(pl.text or "")
                coords=[]; step=2 if tag=="PntList2D" else 3
                for i in range(0,len(nums)-step+1,step):
                    x=nums[i]; y=nums[i+1]; coords.append((x,y))
                if len(coords)>=2: lines.append({"coords": coords, "objtype": None})
    for cg in root.iter():
        if not cg.tag.endswith("CoordGeom"): continue
        objtype_val = None
        pf = None
        for anc in root.iter():
            if anc.tag.endswith("PlanFeature") and cg in list(anc):
                pf = anc; break
        if pf is not None:
            for feat in pf.findall(".//Feature"):
                for prop in feat.findall("Property"):
                    if (prop.get("label") or "").upper() == "OBJTYPE":
                        objtype_val = prop.get("value"); break
                if objtype_val: break
        for geom in list(cg):
            if not geom.tag.endswith("Line"): continue
            stn = geom.find("Start"); enn = geom.find("End")
            if stn is None or enn is None: continue
            svals = _parse_numbers_list(stn.text or ""); evals = _parse_numbers_list(enn.text or "")
            if len(svals)>=2 and len(evals)>=2:
                lines.append({"coords":[(svals[0], svals[1]), (evals[0], evals[1])], "objtype": objtype_val})
    for al in root.iter():
        if not al.tag.endswith("Alignment"): continue
        for cg in al.iter():
            if not cg.tag.endswith("CoordGeom"): continue
            seg=[]
            def get_xy(node):
                if node is None: return None
                ref = node.get("pntRef") if hasattr(node, "get") else None
                if ref and ref in pnt_by_name: return pnt_by_name[ref]
                nums = _parse_numbers_list(getattr(node,"text",None) or "")
                if len(nums)>=2: return (nums[0], nums[1])
                try:
                    x=float(node.get("x")); y=float(node.get("y")); return (x,y)
                except: return None
            for geom in list(cg):
                tag = geom.tag
                if tag.endswith(("Line","Curve","Spiral")):
                    stn = enn = None
                    for ch in list(geom):
                        ctag = ch.tag
                        if ctag.lower().endswith("start"): stn = get_xy(ch)
                        elif ctag.lower().endswith("end"): enn = get_xy(ch)
                    if stn and (not seg or seg[-1]!=stn): seg.append(stn)
                    if enn: seg.append(enn)
            if len(seg)>=2: lines.append({"coords": seg, "objtype": None})
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

    points_df = None; centers_dict=None; centers_df=None
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
                    show_cols = [c for c in [cols["sobj"], cols["east"], cols["north"], cols["alt"], cols["rot"]] if c]
                    st.success(f"Fant {len(points_df)} senterpunkter.")
                    st.dataframe(points_df[show_cols].head(30), use_container_width=True)
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

    st.subheader("Linjer (VA/EL)")
    lines_up = st.file_uploader("Linjer (GeoJSON eller XML/LandXML)", type=["geojson","json","xml","landxml"], key="SB_lines")
    objtype_field = st.text_input("Objekttype-felt i linjefil (valgfritt)", value="objtype", key="SB_objfield")
    epsg_lines = ensure_epsg("SB_epsg_lines", "EPSG for linjer", default=25832)
    type_filter = st.text_input("Typefilter (komma-separert, valgfritt)", value="", key="SB_typefilter")
    buffer_m = st.number_input("Buffer (m) mot linje", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="SB_buffer")
    st.caption("**Avansert (CRS/akse-hjelp):**")
    swap_axes = st.checkbox("Bytt akser X↔Y før reprojeksjon", value=False, key="SB_swap_axes")
    scale_factor = st.number_input("Skalering (f.eks. 0.001 for mm→m)", value=1.0, step=0.001, format="%.6f", key="SB_scale")
    add_e = st.number_input("Legg til E-Offset (meter)", value=0.0, step=1.0, key="SB_addE")
    add_n = st.number_input("Legg til N-Offset (meter)", value=0.0, step=1.0, key="SB_addN")

    lines_list=None
    if lines_up is not None:
        try:
            lines_list = load_lines_auto(lines_up, lines_up.name, prop_objtype=objtype_field or None)
            if type_filter.strip():
                allowed = set([s.strip() for s in type_filter.split(",") if s.strip()])
                lines_list = [L for L in lines_list if (L["objtype"] in allowed)]
            if lines_list:
                n_lines = len(lines_list)
                n_vertices = sum(len(L["coords"]) for L in lines_list)
                some_types = sorted({str(L.get("objtype")) for L in lines_list if L.get("objtype")})[:5]
                st.success(f"Lastet {n_lines} linjer med totalt {n_vertices} punkter. Typer (utdrag): {', '.join(some_types) if some_types else '(ingen oppgitt)'}")
                def adjust_coords(coords):
                    out=[]
                    for (x,y) in coords:
                        ex = y if swap_axes else x
                        ny = x if swap_axes else y
                        ex = ex * scale_factor + add_e
                        ny = ny * scale_factor + add_n
                        out.append((ex,ny))
                    return out
                lines_list = [{"coords": adjust_coords(L["coords"]), "objtype": L.get("objtype")} for L in lines_list]
                allx=[xy[0] for L in lines_list for xy in L["coords"]]
                ally=[xy[1] for L in lines_list for xy in L["coords"]]
                if allx and ally:
                    st.caption(f"Linje-utstrekning (linje-EPSG {epsg_lines} etter justering): E:[{min(allx):.3f}, {max(allx):.3f}] N:[{min(ally):.3f}, {max(ally):.3f}]")
                sample=[]
                for L in lines_list[:3]:
                    if L["coords"]: sample.append(L["coords"][0])
                if sample:
                    st.caption("Eksempel-koordinater (før reproj): " + ", ".join([f"({e:.3f},{n:.3f})" for e,n in sample]))
            else:
                st.warning("Ingen linjer ble tolket fra fila (prøv GeoJSON-eksport eller en annen XML/LandXML).")
        except Exception as e:
            st.exception(e)

    st.subheader("Globale valg")
    draw_arrow_global = st.checkbox("Tegn nordpil på bilder", value=True, key="SB_draw_arrow")
    arrow_size_global = st.slider("Pil-størrelse (px)", 60, 240, 120, key="SB_arrow_size")
    auto_180 = st.checkbox("Auto-180 (flipp heading hvis ~180° fra senter/rotasjon)", value=True, key="SB_auto180")

    st.subheader("Nordpil (farge)")
    arrow_fill = st.color_picker("Fyllfarge", value="#FFFFFF", key="SB_arrow_fill")
    arrow_outline = st.color_picker("Konturfarge", value="#000000", key="SB_arrow_outline")

    if not draw_arrow_global:
        st.info("Nordpil er slått AV i prosjektet – slå på for å tegne pil.")

    st.session_state["POINTS_EPSG"] = epsg_pts
    st.session_state["LINES_EPSG"] = epsg_lines
    st.session_state["BUFFER_M"] = buffer_m
    st.session_state["DRAW_ARROW"] = draw_arrow_global
    st.session_state["ARROW_SIZE"] = arrow_size_global
    st.session_state["AUTO_180"] = auto_180
    st.session_state["ARROW_COLOR"] = hex_to_rgb(arrow_fill, (255,255,255))
    st.session_state["ARROW_OUTLINE"] = hex_to_rgb(arrow_outline, (0,0,0))
    st.session_state["CENTERS_DICT"] = centers_dict
    st.session_state["CENTERS_DF"] = centers_df
    st.session_state["POINTS_DF"] = points_df
    st.session_state["LINES_LIST"] = lines_list
    st.session_state["SWAP_AXES"] = swap_axes
    st.session_state["SCALE_FACTOR"] = scale_factor
    st.session_state["ADD_E"] = add_e
    st.session_state["ADD_N"] = add_n

def heading_from_lines(E,N):
    lines = st.session_state.get("LINES_LIST")
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    epsg_lin = st.session_state.get("LINES_EPSG", 25832)
    buf = st.session_state.get("BUFFER_M", 2.0)
    if not lines: return (None, None)
    if epsg_lin != epsg_pts:
        tr = Transformer.from_crs(epsg_lin, epsg_pts, always_xy=True)
        def reproj(coords):
            return [tuple(tr.transform(x,y)) for x,y in coords]
        use = [{"coords": reproj(L["coords"]), "objtype": L.get("objtype")} for L in lines]
    else:
        use = lines
    best = (None, float("inf"))
    for L in use:
        hd, dist, npt = nearest_heading_on_polyline(L["coords"], (E,N))
        if npt is None: continue
        if dist < best[1]: best=(hd, dist)
    hd, dist = best
    if dist is None: return (None, None)
    return (hd if dist <= buf else None), dist

def choose_pos_and_heading(sobj_label=None, E=None, N=None, Alt=None, Rot=None, manual_override=None):
    centers = st.session_state.get("CENTERS_DICT") or {}
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)

    center_hint = None
    if (E is None or N is None) and sobj_label:
        base = base_id(sobj_label)
        info = centers.get(base)
        if info:
            E = info.get("center_E"); N = info.get("center_N")
            if Alt is None: Alt = info.get("hoyde")
            center_hint = info.get("azimut")

    line_h, dist = heading_from_lines(E, N) if (E is not None and N is not None) else (None, None)
    hd = line_h if _is_valid_number(line_h) else (center_hint if _is_valid_number(center_hint) else (Rot if _is_valid_number(Rot) else None))

    if _is_valid_number(manual_override):
        hd = float(manual_override)

    if st.session_state.get("AUTO_180", True) and not _is_valid_number(manual_override):
        if _is_valid_number(line_h) and _is_valid_number(center_hint):
            d = ang_diff(line_h, center_hint)
            if d is not None and 150 <= d <= 210:
                hd = (line_h + 180) % 360
        elif not _is_valid_number(line_h) and _is_valid_number(Rot) and _is_valid_number(center_hint):
            d = ang_diff(Rot, center_hint)
            if d is not None and 150 <= d <= 210:
                hd = (Rot + 180) % 360

    return E, N, Alt, (_wrap_deg(hd) if _is_valid_number(hd) else None), center_hint, line_h, dist

tabA, tabC, tabD = st.tabs(["A) Batch geotagg", "C) Manuell + 2-klikk", "D) Kart"])

with tabA:
    st.subheader("A) Geotagg mange bilder")
    centers_dict = st.session_state.get("CENTERS_DICT") or {}
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    draw_arrow = st.session_state.get("DRAW_ARROW", True)
    arrow_size = st.session_state.get("ARROW_SIZE", 120)
    arrow_col = st.session_state.get("ARROW_COLOR", (255,255,255))
    arrow_outline = st.session_state.get("ARROW_OUTLINE", (0,0,0))

    picked_label = None
    if centers_dict:
        options = sorted(list(centers_dict.keys()))
        picked_label = st.selectbox("Velg kum/S_OBJID", options, key="A_pick_label")
    else:
        st.warning("Ingen kum-senter i prosjektdata ennå. Last opp punkter i sidepanelet.")

    mode = st.radio("Bildekilde:", ["ZIP-opplasting", "Opplasting (flere filer)"], index=1, key="A_mode")
    exts_ok = (".jpg",".jpeg",".png",".tif",".tiff",".heic",".heif")
    patt = st.selectbox("Nytt filnavn", ["keep","label_orig","label_only","label_en"], index=1, key="A_rename")
    epsg_out = ensure_epsg("TAB_A_DOC_EPSG", "Dokumentasjons-CRS for CSV (eksport)", default=25832)

    with st.expander("Avansert: manuell heading/Nord (grader)"):
        man_enable = st.checkbox("Overstyr heading manuelt", value=False, key="A_manual_on")
        man_heading = st.number_input("Manuell heading (0–359°)", min_value=0, max_value=359, value=0, key="A_manual_val")

    zip_up=None; files_up=None
    if mode=="ZIP-opplasting":
        zip_up = st.file_uploader("Last opp ZIP med bilder", type=["zip"], key="A_zip2")
    else:
        files_up = st.file_uploader("Dra inn flere bilder", type=[e[1:] for e in exts_ok], accept_multiple_files=True, key="A_files2")
        if not HEIC_OK: st.info("HEIC/HEIF krever 'pillow-heif'.")

    def load_any_to_jpeg_bytes(payload: bytes) -> bytes:
        im = Image.open(io.BytesIO(payload))
        im = normalize_orientation(im).convert("RGB")
        buf = io.BytesIO(); im.save(buf,"jpeg", quality=95); return buf.getvalue()

    def build_new_name(pattern, label, orig_name, E=None, N=None):
        base, ext = os.path.splitext(orig_name)
        safe = re.sub(r'[\\/:*?"<>|]+','_', str(label)).strip().replace(" ","_")
        if pattern=="keep" or not safe: return f"{base}.jpg"
        if pattern=="label_orig": return f"{safe}_{base}.jpg"
        if pattern=="label_only": return f"{safe}.jpg"
        if pattern=="label_en":
            e_txt = f"{int(round(E))}" if E is not None else "E"
            n_txt = f"{int(round(N))}" if N is not None else "N"
            return f"{safe}_{e_txt}_{n_txt}.jpg"
        return f"{base}.jpg"

    if st.button("Kjør geotag (Tab A)", key="A_run"):
        try:
            if not picked_label or picked_label not in centers_dict:
                st.error("Velg en kum/S_OBJID i listen.")
            else:
                info = centers_dict[picked_label]
                E0 = info.get("center_E"); N0 = info.get("center_N"); Alt0 = info.get("hoyde")
                if E0 is None or N0 is None:
                    st.error(f"Kum '{picked_label}' mangler E/N.")
                else:
                    processed=[]; skipped=[]
                    zout_mem = io.BytesIO(); zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)
                    used=set(); counter={"n":0}
                    def process_one(name, payload):
                        name = str(name)
                        if not name.lower().endswith(exts_ok): 
                            skipped.append({"file": name, "reason":"Ikke-støttet filtype"}); return
                        try: jpeg0 = load_any_to_jpeg_bytes(payload)
                        except Exception as e:
                            skipped.append({"file": name, "reason": f"Kunne ikke lese bilde: {e}"}); return
                        E,N,Alt,hd,cent,line_h,dist = choose_pos_and_heading(picked_label, E0,N0, Alt0, None, manual_override=(float(man_heading) if man_enable else None))
                        if E is None or N is None:
                            skipped.append({"file": name, "reason":"Mangler E/N"}); return
                        lat, lon = transform_EN_to_wgs84(E, N, epsg_pts)
                        im = Image.open(io.BytesIO(jpeg0))
                        use_hd = hd if _is_valid_number(hd) else (float(man_heading) if man_enable else None)
                        if draw_arrow and _is_valid_number(use_hd):
                            im = draw_north_arrow(im, _wrap_deg(use_hd), size_px=arrow_size, color=st.session_state.get("ARROW_COLOR",(255,255,255)), outline=st.session_state.get("ARROW_OUTLINE",(0,0,0)))
                        buf = io.BytesIO(); im.save(buf,"jpeg", quality=95); buf.seek(0)
                        jpeg1 = write_exif_jpeg_bytes(buf.getvalue(), lat, lon, Alt, use_hd)
                        Xdoc,Ydoc = transform_EN_to_epsg(E,N, epsg_pts, epsg_out)

                        if _is_valid_number(line_h):
                            heading_source = "linje"
                        elif _is_valid_number(cent):
                            heading_source = "kum-azimut"
                        elif _is_valid_number(use_hd):
                            heading_source = "manuell"
                        else:
                            heading_source = "ukjent"

                        processed.append({
                            "file": name,
                            "S_OBJID": picked_label,
                            "E_in": E, "N_in": N,
                            "lat": lat, "lon": lon,
                            f"E_{epsg_out}": Xdoc, f"N_{epsg_out}": Ydoc,
                            "hoyde": Alt,
                            "heading": use_hd,
                            "heading_line": line_h,
                            "center_hint": cent,
                            "dist_to_line": dist,
                            "heading_source": heading_source
                        })
                        newname = build_new_name(patt, picked_label, os.path.basename(name), E0, N0)
                        base,ext=os.path.splitext(newname); cand=newname; i=1
                        while cand in used: cand=f"{base}_{i}.jpg"; i+=1
                        used.add(cand); zout.writestr(cand, jpeg1); counter["n"]+=1

                    if mode=="ZIP-opplasting":
                        if not zip_up: st.error("Last opp ZIP.")
                        else:
                            zin = zipfile.ZipFile(io.BytesIO(zip_up.read()), "r")
                            for name in zin.namelist():
                                if not name.lower().endswith(exts_ok): continue
                                process_one(os.path.basename(name), zin.read(name))
                    else:
                        if not files_up or len(files_up)==0: st.error("Dra inn bilder.")
                        else:
                            for f in files_up: process_one(f.name, f.read())

                    zout.close()
                    if counter["n"]>0:
                        st.download_button("Last ned ZIP", data=zout_mem.getvalue(), file_name="geotagged.zip", mime="application/zip")
                        st.success(f"Geotagget {counter['n']} bilder.")
                    else:
                        st.error("Ingen bilder skrevet (0).")

                    if processed:
                        dfo=pd.DataFrame(processed)
                        st.markdown("**Debug: første 10 rader (heading-kilde, dist_to_line m.m.)**")
                        st.dataframe(dfo.head(10), use_container_width=True)
                        st.download_button("Last ned CSV (Tab A)", dfo.to_csv(index=False).encode("utf-8"), "geotag_tabA.csv", "text/csv")
                    if skipped:
                        st.warning("Hoppet over:"); st.dataframe(pd.DataFrame(skipped))
        except Exception as e:
            st.exception(e)

with tabC:
    st.subheader("C) Manuell pr. bilde + to-klikk-orientering")
    centers_dict = st.session_state.get("CENTERS_DICT") or {}
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    draw_arrow = st.session_state.get("DRAW_ARROW", True)
    arrow_size = st.session_state.get("ARROW_SIZE", 120)
    arrow_col = st.session_state.get("ARROW_COLOR", (255,255,255))
    arrow_outline = st.session_state.get("ARROW_OUTLINE", (0,0,0))

    if centers_dict:
        options = sorted(list(centers_dict.keys()))
        picked_label_C = st.selectbox("Velg kum/S_OBJID", options, key="C_pick_label")
    else:
        picked_label_C=None; st.warning("Last opp punkter i sidepanelet.")

    files_up_C = st.file_uploader("Dra inn bilder (flere)", type=["jpg","jpeg","png","tif","tiff","heic","heif"], accept_multiple_files=True, key="C_files")
    if files_up_C and len(files_up_C)>0 and picked_label_C:
        if "MANUAL_HEADINGS" not in st.session_state: st.session_state["MANUAL_HEADINGS"]={}
        man_dict = st.session_state["MANUAL_HEADINGS"]
        names=[f.name for f in files_up_C]
        sel = st.selectbox("Velg bilde", names, key="C_sel_name")
        cur_idx = names.index(sel) if sel in names else 0

        colL, colR = st.columns([2,1])
        with colL:
            f = files_up_C[cur_idx]; payload = f.read(); f.seek(0)
            im0 = Image.open(io.BytesIO(payload)); im0 = normalize_orientation(im0).convert("RGB")
            info = centers_dict.get(picked_label_C, {})
            E0 = info.get("center_E"); N0 = info.get("center_N"); Alt0 = info.get("hoyde")
            E,N,Alt,hd,cent,line_h,dist = choose_pos_and_heading(picked_label_C, E0,N0, Alt0, None, manual_override=None)
            cur_manual = man_dict.get(sel)
            show_hd = cur_manual if _is_valid_number(cur_manual) else hd
            if draw_arrow and _is_valid_number(show_hd):
                im_prev = draw_north_arrow(im0.copy(), _wrap_deg(show_hd), size_px=arrow_size, color=arrow_col, outline=arrow_outline)
            else:
                im_prev = im0
            st.image(im_prev, caption=f"Forhåndsvisning – heading={show_hd if show_hd is not None else '—'}°", use_column_width=True)

            with st.expander("Orienter med 2 hjørner (klikk i bildet)"):
                pts_df = st.session_state.get("POINTS_DF")
                delims_val = st.session_state.get("SB_delims", "-_ ./") if "SB_delims" in st.session_state else "-_ ./"
                base_lbl = base_id(picked_label_C, delims_val) if picked_label_C else None
                corners_df = None
                if pts_df is not None and base_lbl:
                    cols = detect_columns(pts_df)
                    if cols["sobj"] and cols["east"] and cols["north"]:
                        tmp = pts_df.copy()
                        tmp["_base"] = tmp[cols["sobj"]].astype(str).map(lambda s: base_id(s, delims_val))
                        grp = tmp[tmp["_base"] == base_lbl]
                        if len(grp) >= 2:
                            corners_df = grp.rename(columns={cols["east"]:"E", cols["north"]:"N"})
                            corners_df = corners_df.reset_index(drop=True)

                if corners_df is None or len(corners_df) < 2:
                    st.info("Finner ikke minst 2 hjørner for denne kummen i opplastet punktfil.")
                else:
                    corners_df = corners_df.reset_index(drop=True).copy()
                    corners_df["idx"] = corners_df.index
                    show_cols = ["idx"]
                    if "S_OBJID" in corners_df.columns: show_cols.append("S_OBJID")
                    show_cols += [c for c in ["E","N"] if c in corners_df.columns]
                    st.dataframe(corners_df[show_cols].head(30), use_container_width=True)

                    st.caption("Klikk først Hjørne A, deretter Hjørne B i bildet:")
                    click_key = f"C_clicks_{picked_label_C}_{sel}"
                    coords = img_coords(im0, key=click_key)
                    if coords:
                        clicks = st.session_state.get(click_key + "_list", [])
                        if not clicks or (clicks and (clicks[-1] != (coords['x'], coords['y']))):
                            clicks.append((coords["x"], coords["y"]))
                            clicks = clicks[-2:]
                            st.session_state[click_key + "_list"] = clicks
                    clicks = st.session_state.get(click_key + "_list", [])
                    st.write(f"Klikk: {clicks}")

                    idxA = st.number_input("Indeks hjørne A (radnr 0..)", min_value=0, max_value=len(corners_df)-1, value=0, key="C_cA_idx")
                    idxB = st.number_input("Indeks hjørne B (radnr 0..)", min_value=0, max_value=len(corners_df)-1, value=min(1, len(corners_df)-1), key="C_cB_idx")
                    EA = parse_float_maybe_comma(corners_df.loc[idxA, "E"]) if 0 <= idxA < len(corners_df) else None
                    NA = parse_float_maybe_comma(corners_df.loc[idxA, "N"]) if 0 <= idxA < len(corners_df) else None
                    EB = parse_float_maybe_comma(corners_df.loc[idxB, "E"]) if 0 <= idxB < len(corners_df) else None
                    NB = parse_float_maybe_comma(corners_df.loc[idxB, "N"]) if 0 <= idxB < len(corners_df) else None

                    if _is_valid_number(EA) and _is_valid_number(NA) and _is_valid_number(EB) and _is_valid_number(NB):
                        if len(clicks) == 2:
                            (xA,yA),(xB,yB) = clicks
                            az_real = (math.degrees(math.atan2(EB - EA, NB - NA)) + 360.0) % 360.0
                            az_img  = (math.degrees(math.atan2(xB - xA, -(yB - yA))) + 360.0) % 360.0
                            hd2 = (az_real - az_img) % 360.0
                            st.success(f"Beregnet heading = {hd2:.1f}° (lagres som manuell)")
                            st.session_state["MANUAL_HEADINGS"][sel] = float(hd2)
                            im_prev2 = draw_north_arrow(im0.copy(), _wrap_deg(hd2), size_px=arrow_size, color=arrow_col, outline=arrow_outline)
                            st.image(im_prev2, caption=f"Forhåndsvisning (2-klikk) = {hd2:.1f}°", use_column_width=True)
                        if st.button("Nullstill 2-klikk", key="C_clearclicks_btn"):
                            st.session_state[click_key + "_list"] = []
                    else:
                        st.warning("Ugyldige E/N for valgte hjørner.")

        with colR:
            st.markdown("**Sett heading for valgt bilde**")
            if not _is_valid_number(hd): st.info("Ingen heading funnet automatisk – sett manuelt eller bruk 2-klikk.")
            cur_manual = st.session_state["MANUAL_HEADINGS"].get(sel)
            base_val = int(cur_manual if _is_valid_number(cur_manual) else int(hd or 0))
            man_val = st.slider("Manuell heading (0–359°)", 0, 359, base_val, key="C_slider")
            c1,c2,c3 = st.columns(3)
            if c1.button("−10°", key="C_m10"): man_val = (man_val-10)%360; st.session_state["C_slider"]=man_val
            if c2.button("+10°", key="C_p10"): man_val = (man_val+10)%360; st.session_state["C_slider"]=man_val
            if c3.button("Flip 180°", key="C_flip"): man_val = (man_val+180)%360; st.session_state["C_slider"]=man_val
            if st.button("Lagre heading", key="C_save_one"):
                st.session_state["MANUAL_HEADINGS"][sel]=float(man_val); st.success(f"Lagret {sel}: {man_val}°")

        st.markdown("---")
        st.markdown("**Kart (kum-senter + heading-vektor + hjørnepunkter)**")
        try:
            import pydeck as pdk
            if E0 is not None and N0 is not None:
                show_corners = st.checkbox("Vis hjørnepunkter i kartet", value=True, key="C_show_corners")
                corner_size = st.slider("Størrelse på hjørnepunkter", 1, 20, 4, key="C_corner_size")
                line_width = st.slider("Linjebredde (VA/EL-linjer)", 1, 8, 2, key="C_line_width")
                label_mode = st.selectbox("Hjørne-etikett", ["idx", "punktnavn", "ingen"], index=0, key="C_corner_label")

                lat0, lon0 = transform_EN_to_wgs84(E0, N0, epsg_pts)
                use_heading = st.session_state["MANUAL_HEADINGS"].get(sel, show_hd)
                layers=[]
                centers_df = st.session_state.get("CENTERS_DF")
                if centers_df is not None and not centers_df.empty:
                    tr_pts = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
                    tmp = centers_df.copy()
                    tmp["lon"], tmp["lat"] = zip(*[tr_pts.transform(e, n) for e,n in zip(tmp["center_E"], tmp["center_N"])])
                    tmp["color"] = [[0,150,255]]*len(tmp)
                    layers.append(pdk.Layer("ScatterplotLayer", tmp, get_position='[lon, lat]', get_radius=4, get_fill_color='color', pickable=True))
                    try:
                        layers.append(pdk.Layer("TextLayer", tmp, get_position='[lon, lat]', get_text="base_id", get_size=12, get_color=[0,0,0], get_angle=0, get_alignment_baseline="bottom"))
                    except Exception: pass
                lines = st.session_state.get("LINES_LIST")
                epsg_lin = st.session_state.get("LINES_EPSG", 25832)
                if lines:
                    if epsg_lin != 4326:
                        tr_lin = Transformer.from_crs(epsg_lin, 4326, always_xy=True)
                        def to_wgs_path(coords): return [[*tr_lin.transform(x,y)] for (x,y) in coords]
                    else:
                        def to_wgs_path(coords): return [[x,y] for (x,y) in coords]
                    paths = [{"path": to_wgs_path(L["coords"])} for L in lines]
                    layers.append(pdk.Layer("PathLayer", paths, get_path="path", get_width=line_width, get_color=[80,80,200]))
                if show_corners:
                    pts_all = st.session_state.get("POINTS_DF")
                    delims_val2 = st.session_state.get("SB_delims", "-_ ./") if "SB_delims" in st.session_state else "-_ ./"
                    if pts_all is not None and picked_label_C:
                        cols_all = detect_columns(pts_all)
                        if cols_all["sobj"] and cols_all["east"] and cols_all["north"]:
                            tmpP = pts_all.copy()
                            tmpP["_base"] = tmpP[cols_all["sobj"]].astype(str).map(lambda s: base_id(s, delims_val2))
                            base_lbl2 = base_id(picked_label_C, delims_val2)
                            grp = tmpP[tmpP["_base"] == base_lbl2].reset_index(drop=True).copy()
                            if len(grp) >= 1:
                                tr_tmp = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
                                def _to_xy(e,n):
                                    e2 = parse_float_maybe_comma(e); n2 = parse_float_maybe_comma(n)
                                    if e2 is None or n2 is None: return None
                                    x,y = tr_tmp.transform(e2, n2); return (x,y)
                                ll = [ _to_xy(e,n) for e,n in zip(grp[cols_all["east"]], grp[cols_all["north"]]) ]
                                grp = grp.assign(lon=[p[0] if p else None for p in ll], lat=[p[1] if p else None for p in ll])
                                grp = grp.dropna(subset=["lon","lat"]).reset_index(drop=True)
                                grp["idx"] = grp.index
                                sobj_col = cols_all["sobj"] if cols_all["sobj"] else None
                                if label_mode == "punktnavn" and sobj_col and sobj_col in grp.columns:
                                    grp["_label"] = grp[sobj_col].astype(str)
                                elif label_mode == "idx":
                                    grp["_label"] = grp["idx"].astype(str)
                                else:
                                    grp["_label"] = ""
                                grp["color"] = [[0,255,0]]*len(grp)
                                layers.append(pdk.Layer("ScatterplotLayer", grp, get_position='[lon, lat]', get_radius=corner_size, get_fill_color='color', pickable=True))
                                try:
                                    if label_mode != "ingen":
                                        layers.append(pdk.Layer("TextLayer", grp, get_position='[lon, lat]', get_text="_label", get_size=12, get_color=[0,120,0], get_angle=0, get_alignment_baseline="top"))
                                except Exception: pass
                if _is_valid_number(use_heading):
                    L=5.0; rad=math.radians(use_heading)
                    E1 = E0 + math.sin(rad)*L; N1 = N0 + math.cos(rad)*L
                    lat1, lon1 = transform_EN_to_wgs84(E1, N1, epsg_pts)
                    arrow_path=[{"path":[[lon0,lat0],[lon1,lat1]]}]
                    layers.append(pdk.Layer("PathLayer", arrow_path, get_path="path", get_width=3, get_color=[200,60,60]))
                view_state = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=18)
                st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view_state), use_container_width=True)
            else:
                st.info("Kum-senter mangler E/N – last opp punkter.")
        except Exception as e:
            st.info("Kunne ikke vise kart (pydeck mangler eller feil).")

        st.markdown("---")
        if st.button("Eksporter alle som ZIP (med manuell heading der satt)", key="C_export"):
            processed=0; skipped=[]; zout_mem=io.BytesIO(); zout=zipfile.ZipFile(zout_mem,"w",zipfile.ZIP_DEFLATED)
            for f in files_up_C:
                try:
                    payload=f.read(); f.seek(0)
                    im0 = Image.open(io.BytesIO(payload)); im0 = normalize_orientation(im0).convert("RGB")
                    info = centers_dict.get(picked_label_C, {})
                    E0 = info.get("center_E"); N0 = info.get("center_N"); Alt0 = info.get("hoyde")
                    man = st.session_state["MANUAL_HEADINGS"].get(f.name)
                    if man is None and f.name == sel:
                        man = st.session_state["MANUAL_HEADINGS"].get(sel)
                    E,N,Alt,hd,cent,line_h,dist = choose_pos_and_heading(picked_label_C, E0,N0, Alt0, None, manual_override=man if man is not None else None)
                    if E is None or N is None: skipped.append({"file":f.name,"reason":"Mangler E/N"}); continue
                    lat, lon = transform_EN_to_wgs84(E, N, epsg_pts)
                    if draw_arrow and _is_valid_number(hd):
                        im0 = draw_north_arrow(im0, _wrap_deg(hd), size_px=arrow_size, color=arrow_col, outline=arrow_outline)
                    buf=io.BytesIO(); im0.save(buf,"jpeg", quality=95); buf.seek(0)
                    jpeg1 = write_exif_jpeg_bytes(buf.getvalue(), lat, lon, Alt, hd)
                    newname = f"{picked_label_C}_{os.path.splitext(os.path.basename(f.name))[0]}.jpg"
                    zout.writestr(newname, jpeg1); processed+=1
                except Exception as e:
                    skipped.append({"file": f.name, "reason": str(e)})
            zout.close()
            if processed>0:
                st.download_button("Last ned ZIP (Tab C)", data=zout_mem.getvalue(), file_name="geotag_manual.zip", mime="application/zip")
                st.success(f"Skrev {processed} bilder.")
            if skipped:
                st.warning("Hoppet over:"); st.dataframe(pd.DataFrame(skipped))
    else:
        st.info("Last opp bilder og velg kum for manuell forhåndsvisning.")

with tabD:
    st.subheader("D) Kart – senterpunkter og linjer")
    line_width_D = st.slider("Linjebredde (oversiktskart)", 1, 8, 1, key="D_line_width")
    centers_df = st.session_state.get("CENTERS_DF")
    lines = st.session_state.get("LINES_LIST")
    try:
        import pydeck as pdk
        epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
        epsg_lin = st.session_state.get("LINES_EPSG", 25832)
        layers=[]
        if centers_df is not None and not centers_df.empty:
            tr_pts = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
            tmp = centers_df.copy()
            tmp["lon"], tmp["lat"] = zip(*[tr_pts.transform(e, n) for e,n in zip(tmp["center_E"], tmp["center_N"])])
            tmp["color"] = [[0,150,255]]*len(tmp)
            layers.append(pdk.Layer("ScatterplotLayer", tmp, get_position='[lon, lat]', get_radius=4, get_fill_color='color', pickable=True))
            try:
                layers.append(pdk.Layer("TextLayer", tmp, get_position='[lon, lat]', get_text="base_id", get_size=12, get_color=[0,0,0], get_angle=0, get_alignment_baseline="bottom"))
            except Exception: pass
            view_state = pdk.ViewState(latitude=float(tmp["lat"].mean()), longitude=float(tmp["lon"].mean()), zoom=16)
        else:
            view_state = pdk.ViewState(latitude=59.91, longitude=10.75, zoom=10)

        if lines:
            if epsg_lin != 4326:
                tr_lin = Transformer.from_crs(epsg_lin, 4326, always_xy=True)
                def to_wgs_path(coords): return [[*tr_lin.transform(x,y)] for (x,y) in coords]
            else:
                def to_wgs_path(coords): return [[x,y] for (x,y) in coords]
            paths = [{"path": to_wgs_path(L["coords"])} for L in lines]
            layers.append(pdk.Layer("PathLayer", paths, get_path="path", get_width=line_width_D, get_color=[80,80,200]))
        else:
            st.info("Ingen linjer lastet eller tolket fra filen.")

        st.pydeck_chart(pdk.Deck(map_style=None, layers=layers, initial_view_state=view_state), use_container_width=True)
    except Exception as e:
        st.info("Kunne ikke vise kart (pydeck mangler eller feil).")

st.markdown("---")
st.caption("v11.3 • Hjørneetiketter og linjebredde • EXIF WGS84, LandXML/GeoJSON, manuell heading, 2‑klikk, CRS‑verktøy")

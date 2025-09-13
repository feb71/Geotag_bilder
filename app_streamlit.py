# app.py — Geotagging bilder v11.8 (full)
# ------------------------------------------------------------
# Funksjoner:
# - Sidepanel: last inn punkter (hjørner/senter), linjer (GeoJSON, LandXML), CRS, kalibrering
# - Tab A: batch geotagging av en mappe/ZIP med bilder for valgt kum (S_OBJID)
# - Tab B: "målebok" — les punkttabell + S_HYPERLINK (bildefilnavn) og bilder, geotagg automatisk
# - Tab C: manuell orientering pr. bilde:
#     - Klikk i kartet ELLER tegn/drag polyline/marker (Folium Draw) for å sette heading
#     - 2-klikk i selve bildet med to kjente hjørner + velg tilsvarende hjørner i tabell => heading
# - Tab D: oversiktskart (pydeck) med kumsenter, hjørner og linjer, med etiketter
# - Tegner nordpil i bildet (justerbar størrelse + N-tekst-størrelse og farger)
# - Skriver EXIF (WGS84/lat-lon + GPSImgDirection)
# - Automatisk heading fra nærmeste linje (innen buffer), eller kum-azimut, eller rotasjon i tabell
# - Kalibrering: speil (360-θ) + offset (°)
# - CRS-nyttig: bytte akser, skalering, offset før reprojeksjon av linjer
# ------------------------------------------------------------

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

# HEIC/HEIF støtte
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

st.set_page_config(page_title="Geotagging bilder v11.8", layout="wide")
st.title("Geotagging bilder v11.8")
st.caption("v11.8 • Kart-rotasjon (tegn/drag) • justerbar 'N'-størrelse • Tab B (målebok) • etiketter for hjørner/senter • EXIF WGS84")

# ------------------------- Hjelpefunksjoner -------------------------

def deg_to_dms_rational(dd: float):
    sign = 1 if dd >= 0 else -1
    dd = abs(dd)
    d = int(dd)
    m_full = (dd - d) * 60
    m = int(m_full)
    s = round((m_full - m) * 60 * 10000)
    return sign, ((d, 1), (m, 1), (s, 10000))

def _is_valid_number(x):
    try:
        fx = float(x)
    except:
        return False
    return not (math.isnan(fx) or math.isinf(fx))

def _wrap_deg(d):
    d = float(d) % 360.0
    if d < 0:
        d += 360.0
    return d

def ang_diff(a, b):
    if a is None or b is None:
        return None
    return abs((a - b + 180) % 360 - 180)

def exif_gps(lat, lon, alt=None, direction=None):
    sign_lat, lat_dms = deg_to_dms_rational(lat)
    sign_lon, lon_dms = deg_to_dms_rational(lon)
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b'N' if sign_lat >= 0 else b'S',
        piexif.GPSIFD.GPSLatitude: lat_dms,
        piexif.GPSIFD.GPSLongitudeRef: b'E' if sign_lon >= 0 else b'W',
        piexif.GPSIFD.GPSLongitude: lon_dms,
    }
    if _is_valid_number(alt):
        a = float(alt)
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if a >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = (int(round(abs(a) * 100)), 100)
    if _is_valid_number(direction):
        d = _wrap_deg(direction)
        gps[piexif.GPSIFD.GPSImgDirectionRef] = b'T'
        gps[piexif.GPSIFD.GPSImgDirection] = (int(round(d * 100)), 100)
    return gps

def write_exif_jpeg_bytes(jpeg_bytes, lat, lon, alt=None, direction=None):
    # Sikker måte å skrive EXIF tilbake på
    open("tmp_in.jpg", "wb").write(jpeg_bytes)
    try:
        im = Image.open("tmp_in.jpg")
        try:
            exif_dict = piexif.load(im.info.get("exif", b""))
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
        exif_dict["GPS"] = exif_gps(lat, lon, alt, direction)
        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
        exif_bytes = piexif.dump(exif_dict)
        im.save("tmp_out.jpg", "jpeg", exif=exif_bytes, quality=95)
        return open("tmp_out.jpg", "rb").read()
    finally:
        for p in ("tmp_in.jpg", "tmp_out.jpg"):
            try:
                os.remove(p)
            except:
                pass

def normalize_orientation(im: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(im)

def parse_float_maybe_comma(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(" ", "").replace("\xa0", "")
    if s == "" or s.lower() in {"nan", "none", "-"}:
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
        st.info(f"Ingen EPSG valgt – bruker default {default}.")
        epsg = default
    try:
        CRS.from_epsg(epsg)
    except Exception:
        st.error(f"Ugyldig EPSG: {epsg}. Bruker {default}.")
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

def _load_font(size_px: int):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size_px)
            except:
                pass
    return ImageFont.load_default()

def draw_north_arrow(
    im: Image.Image,
    heading_deg: float,
    pos=("right", "bottom"),
    size_px=120,
    margin=20,
    color=(255, 255, 255),
    outline=(0, 0, 0),
    n_label_size=18,
) -> Image.Image:
    if heading_deg is None:
        return im
    w, h = im.size
    cx = {"left": margin + size_px, "center": w // 2, "right": w - margin - size_px}.get(pos[0], w - margin - size_px)
    cy = {"top": margin + size_px, "middle": h // 2, "bottom": h - margin - size_px}.get(pos[1], h - margin - size_px)
    ang = math.radians(heading_deg % 360.0)
    dx, dy = 0, -1  # "rett opp" i bilde
    vx = dx * math.cos(ang) - dy * math.sin(ang)
    vy = dx * math.sin(ang) + dy * math.cos(ang)
    tip = (int(cx + vx * size_px), int(cy + vy * size_px))
    tail = (int(cx - vx * (size_px * 0.4)), int(cy - vy * (size_px * 0.4)))
    draw = ImageDraw.Draw(im, "RGBA")
    draw.line([tail, tip], fill=color, width=max(4, size_px // 15))
    head_len = size_px * 0.25
    left_ang = math.atan2(vy, vx) + math.radians(150)
    right_ang = math.atan2(vy, vx) - math.radians(150)
    left_pt = (int(tip[0] + math.cos(left_ang) * head_len), int(tip[1] + math.sin(left_ang) * head_len))
    right_pt = (int(tip[0] + math.cos(right_ang) * head_len), int(tip[1] + math.sin(right_ang) * head_len))
    draw.polygon([tip, left_pt, right_pt], fill=color, outline=outline)
    # N-etikett
    try:
        font = _load_font(max(10, int(n_label_size)))
        tx, ty = tip[0] + 8, tip[1] - int(size_px * 0.2)
        draw.text((tx, ty), "N", fill=color, font=font, stroke_width=1, stroke_fill=outline)
    except Exception:
        pass
    return im

def hex_to_rgb(s, default=(255, 255, 255)):
    try:
        s = str(s).strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 3:
            s = "".join([c * 2 for c in s])
        if len(s) != 6:
            return default
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except:
        return default

# Kalibrering
def apply_heading_calibration(theta):
    if theta is None:
        return None
    t = float(theta)
    if st.session_state.get("SB_flip_lr", False):
        t = (360.0 - t) % 360.0
    off = float(st.session_state.get("SB_theta_off", 0.0) or 0.0)
    t = (t + off) % 360.0
    return t

def detect_columns(df: pd.DataFrame):
    low = {c.lower(): c for c in df.columns}
    col = lambda *names: next((low[n] for n in names if n in low), None)
    return {
        "east": col("øst", "oest", "east", "x"),
        "north": col("nord", "north", "y"),
        "alt": col("høyde", "hoyde", "h", "z", "altitude"),
        "rot": col("rotasjon", "retning", "dir", "heading", "azimut", "azimuth"),
        "sobj": col("s_objid", "sobjid", "objid", "obj_id", "punktid", "punkt", "id", "kum"),
        "hyper": col("s_hyperlink", "hyperlink", "bilder", "filer", "photos"),
    }

def dataframe_from_upload(file):
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, dtype=str)
    else:
        return pd.read_csv(file, dtype=str)

def base_id(s: str, delims="-_ ./"):
    s = str(s).strip()
    for d in delims:
        if d in s:
            return s.split(d, 1)[0].strip().upper()
    return s.upper()

def _parse_numbers_list(txt):
    if not txt:
        return []
    parts = str(txt).strip().replace(",", " ").split()
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            pass
    return vals

# -------------------- Linjeleser: LandXML & GeoJSON --------------------

def _parse_numbers_list(txt):
    import re
    if not txt: return []
    parts = re.split(r'[\s,;]+', str(txt).strip())
    vals = []
    for p in parts:
        try: vals.append(float(p))
        except: pass
    return vals

def load_lines_landxml(file_obj):
    """
    Leser LandXML med PlanFeatures/PlanFeature/CoordGeom/Line/Start|End.
    Tåler pos som tekst ('N E Z') eller som attrib (x/y eller X/Y), og pntRef.
    """
    import xml.etree.ElementTree as ET, io as _io
    text = file_obj.read()
    if isinstance(text, bytes):
        # mange Gemini-filer er ISO-8859-1
        try:
            text = text.decode("utf-8")
        except:
            text = text.decode("iso-8859-1", errors="ignore")

    it = ET.iterparse(_io.StringIO(text))
    for _, el in it:
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    root = it.root

    # evt. støtte for <Pnts><Pnt name="...">x y z</Pnt> og pntRef
    pnt_by_name = {}
    for p in root.iter():
        if p.tag.endswith("Pnt") and p.get("name"):
            nums = _parse_numbers_list(p.text or "")
            if len(nums) >= 2:
                pnt_by_name[p.get("name")] = (nums[0], nums[1])

    def get_xy(node):
        if node is None:
            return None
        # 1) pntRef
        ref = node.get("pntRef") if hasattr(node, "get") else None
        if ref and ref in pnt_by_name:
            return pnt_by_name[ref]
        # 2) tekst: "x y [z]" (ofte N E Z i Gemini)
        nums = _parse_numbers_list(getattr(node, "text", None) or "")
        if len(nums) >= 2:
            return (nums[0], nums[1])
        # 3) attrib x/y eller X/Y
        for kx, ky in (("x", "y"), ("X", "Y")):
            if node.get(kx) and node.get(ky):
                try:
                    return (float(node.get(kx)), float(node.get(ky)))
                except:
                    pass
        return None

    lines = []
    for cg in root.iter():
        if not cg.tag.endswith("CoordGeom"):
            continue
        for geom in list(cg):
            if not geom.tag.endswith("Line"):
                continue
            s = get_xy(geom.find("Start"))
            e = get_xy(geom.find("End"))
            if s and e:
                lines.append({"coords": [s, e], "objtype": None})

    # evt. PntList2D/3D
    for tag in ("PntList2D", "PntList3D"):
        for pl in root.iter():
            if pl.tag.endswith(tag):
                nums = _parse_numbers_list(pl.text or "")
                step = 2 if tag == "PntList2D" else 3
                coords = []
                for i in range(0, len(nums) - step + 1, step):
                    coords.append((nums[i], nums[i + 1]))
                if len(coords) >= 2:
                    lines.append({"coords": coords, "objtype": None})

    return lines

def load_lines_gml(file_obj, prop_objtype=None):
    """
    Leser GML (gml:LineString/MultiLineString) m/posList/coordinates.
    """
    import xml.etree.ElementTree as ET, io as _io
    text = file_obj.read()
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except:
            text = text.decode("iso-8859-1", errors="ignore")

    it = ET.iterparse(_io.StringIO(text))
    for _, el in it:
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    root = it.root

    def parse_poslist(txt):
        vals = _parse_numbers_list(txt)
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]

    lines = []
    for ln in root.iter():
        t = ln.tag.lower()
        if t in ("linestring", "multilinestring"):
            for p in ln.iter():
                pt = p.tag.lower()
                if pt == "poslist":
                    coords = parse_poslist(p.text or "")
                    if len(coords) >= 2:
                        lines.append({"coords": coords, "objtype": None})
                elif pt == "coordinates":
                    txt = (p.text or "").replace(",", " ").strip()
                    vals = _parse_numbers_list(txt)
                    coords = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
                    if len(coords) >= 2:
                        lines.append({"coords": coords, "objtype": None})
    return lines

def load_lines_auto(file_obj, filename, prop_objtype=None):
    """
    Prøv i rekkefølge: GeoJSON → GML → LandXML → GeoJSON fallback.
    """
    name = (filename or "").lower()

    # GeoJSON
    if name.endswith((".geojson", ".json")):
        try:
            file_obj.seek(0)
            return load_lines_geojson(file_obj, prop_objtype)
        except:
            pass

    # GML-signatur?
    try:
        file_obj.seek(0)
        head = file_obj.read(4096)
        if isinstance(head, bytes):
            try:
                head = head.decode("utf-8", errors="ignore")
            except:
                head = head.decode("iso-8859-1", errors="ignore")
        if "<gml:" in head or "<LineString" in head or "<MultiLineString" in head:
            file_obj.seek(0)
            return load_lines_gml(file_obj, prop_objtype)
    except:
        pass

    # LandXML
    try:
        file_obj.seek(0)
        return load_lines_landxml(file_obj)
    except:
        pass

    # fallback: forsøk GeoJSON igjen
    try:
        file_obj.seek(0)
        return load_lines_geojson(file_obj, prop_objtype)
    except:
        return []


# ------------------------- Sidepanel (prosjektdata) -------------------------

with st.sidebar:
    st.header("Prosjektdata")
    st.subheader("Punkter / Kummer")
    pts_mode = st.radio("Innhold:", ["Hjørner (beregn senter)", "Senterpunkter (direkte E/N)"], index=0, key="SB_pts_mode", horizontal=True)
    pts_up = st.file_uploader("Excel/CSV for ALLE kummer (S_OBJID, Øst, Nord, (Høyde), (Rotasjon))", type=["xlsx", "xls", "csv"], key="SB_pts")
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
                            A = np.array(points, dtype=float)
                            c = A.mean(axis=0)
                            A0 = A - c
                            U, S, Vt = np.linalg.svd(A0, full_matrices=False)
                            vx, vy = Vt[0, 0], Vt[0, 1]
                            if vy < 0:  # retning opp/nord-ish
                                vx, vy = -vx, -vy
                            az = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0
                            proj_main = A0 @ Vt[0, :]
                            proj_ortho = A0 @ Vt[1, :]
                            L = (proj_main.max() - proj_main.min())
                            W = (proj_ortho.max() - proj_ortho.min())
                            return (float(c[0]), float(c[1])), az, abs(float(L)), abs(float(W))

                        points_df["_base"] = points_df[cols["sobj"]].astype(str).map(lambda s: base_id(s, delims))
                        recs = []
                        for base, grp in points_df.groupby("_base"):
                            pts = []
                            zs = []
                            for _, r in grp.iterrows():
                                e = parse_float_maybe_comma(r[cols["east"]])
                                n = parse_float_maybe_comma(r[cols["north"]])
                                if e is None or n is None:
                                    continue
                                pts.append((e, n))
                                if cols["alt"]:
                                    z = parse_float_maybe_comma(r[cols["alt"]])
                                    if z is not None:
                                        zs.append(z)
                            if len(pts) >= 3:
                                (ce, cn), az, L, W = pca_heading(pts)
                                zmed = float(pd.Series(zs).median()) if len(zs) > 0 else None
                                recs.append({"base_id": base, "center_E": ce, "center_N": cn, "azimut": az, "hoyde": zmed, "count": len(pts)})
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
    lines_up = st.file_uploader("Linjer (GeoJSON eller XML/LandXML)", type=["geojson", "json", "xml", "landxml"], key="SB_lines")
    objtype_field = st.text_input("Objekttype-felt i linjefil (valgfritt)", value="objtype", key="SB_objfield")
    epsg_lines = ensure_epsg("SB_epsg_lines", "EPSG for linjer", default=25832)
    type_filter = st.text_input("Typefilter (komma-separert, valgfritt)", value="", key="SB_typefilter")
    buffer_m = st.number_input("Buffer (m) mot linje", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="SB_buffer")

    st.caption("**Avansert (CRS/akse-hjelp):**")
    swap_axes = st.checkbox("Bytt akser X↔Y før reprojeksjon", value=False, key="SB_swap_axes")
    scale_factor = st.number_input("Skalering (f.eks. 0.001 for mm→m)", value=1.0, step=0.001, format="%.6f", key="SB_scale")
    add_e = st.number_input("Legg til E-Offset (meter)", value=0.0, step=1.0, key="SB_addE")
    add_n = st.number_input("Legg til N-Offset (meter)", value=0.0, step=1.0, key="SB_addN")

    lines_list = None
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

                # etter lines_list = load_lines_auto(...)

                auto_swap = st.checkbox("Auto-oppdag NE→EN (prøv)", value=True, key="SB_auto_swap")

                if auto_swap and lines_list:
                # se på noen første punkter; hvis "x" ser ut som Nord (millioner) og "y" er ~100k, bytt
                    import numpy as np
                    sample = [xy for L in lines_list for xy in L["coords"][:5]]
                    xs = np.array([p[0] for p in sample], dtype=float)
                    ys = np.array([p[1] for p in sample], dtype=float)
                    if np.nanmedian(xs) > np.nanmedian(ys) * 3:  # enkel terskel
                        st.info("Oppdaget NE-rekkefølge → bytter akser automatisk.")
                        swap_axes = True

                
                def adjust_coords(coords):
                    out = []
                    for (x, y) in coords:
                        ex = y if swap_axes else x
                        ny = x if swap_axes else y
                        ex = ex * scale_factor + add_e
                        ny = ny * scale_factor + add_n
                        out.append((ex, ny))
                    return out

                lines_list = [{"coords": adjust_coords(L["coords"]), "objtype": L.get("objtype")} for L in lines_list]
                allx = [xy[0] for L in lines_list for xy in L["coords"]]
                ally = [xy[1] for L in lines_list for xy in L["coords"]]
                if allx and ally:
                    st.caption(f"Linje-utstrekning (linje-EPSG {epsg_lines} etter justering): E:[{min(allx):.3f}, {max(allx):.3f}] N:[{min(ally):.3f}, {max(ally):.3f}]")
            else:
                st.warning("Ingen linjer ble tolket fra fila (prøv GeoJSON-eksport eller en annen XML/LandXML).")
        except Exception as e:
            st.exception(e)


    st.subheader("Globale valg")
    draw_arrow_global = st.checkbox("Tegn nordpil på bilder", value=True, key="SB_draw_arrow")
    arrow_size_global = st.slider("Pil-størrelse (px)", 60, 240, 120, key="SB_arrow_size")
    n_label_size = st.slider("'N'-tekst (px)", 8, 64, 20, key="SB_n_label_size")
    auto_180 = st.checkbox("Auto-180 (flipp heading hvis ~180° fra senter/rotasjon)", value=True, key="SB_auto180")

    st.subheader("Nordpil (farge)")
    arrow_fill = st.color_picker("Fyllfarge", value="#FFFFFF", key="SB_arrow_fill")
    arrow_outline = st.color_picker("Konturfarge", value="#000000", key="SB_arrow_outline")

    st.subheader("Heading-kalibrering")
    flip_lr = st.checkbox("Speilvend retning (360°−θ)", value=False, key="SB_flip_lr")
    theta_off = st.number_input("Heading-justering (°)", min_value=-180.0, max_value=180.0, value=0.0, step=0.5, key="SB_theta_off")

    if not draw_arrow_global:
        st.info("Nordpil er slått AV i prosjektet – slå på for å tegne pil.")

    # Lagre i session_state
    st.session_state["POINTS_EPSG"] = epsg_pts
    st.session_state["LINES_EPSG"] = epsg_lines
    st.session_state["BUFFER_M"] = buffer_m
    st.session_state["DRAW_ARROW"] = draw_arrow_global
    st.session_state["ARROW_SIZE"] = arrow_size_global
    st.session_state["N_LABEL_SIZE"] = n_label_size
    st.session_state["AUTO_180"] = auto_180
    st.session_state["ARROW_COLOR"] = hex_to_rgb(arrow_fill, (255, 255, 255))
    st.session_state["ARROW_OUTLINE"] = hex_to_rgb(arrow_outline, (0, 0, 0))
    st.session_state["CENTERS_DICT"] = centers_dict
    st.session_state["CENTERS_DF"] = centers_df
    st.session_state["POINTS_DF"] = points_df
    st.session_state["LINES_LIST"] = lines_list
    st.session_state["SWAP_AXES"] = swap_axes
    st.session_state["SCALE_FACTOR"] = scale_factor
    st.session_state["ADD_E"] = add_e
    st.session_state["ADD_N"] = add_n

# ------------------------- Felles heading/posisjon -------------------------


def nearest_heading_on_polyline(coords, pt):
    """
    coords: liste [(E,N), ...]
    pt: (E,N)
    return: (heading_deg, dist, (projE, projN))
    """
    if not coords or len(coords) < 2 or pt is None:
        return (None, None, None)

    px, py = pt
    best_hd = None
    best_dist = float("inf")
    best_proj = None

    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        vx, vy = (x2 - x1), (y2 - y1)
        L2 = vx * vx + vy * vy
        if L2 <= 0:
            continue

        wx, wy = (px - x1), (py - y1)
        t = (vx * wx + vy * wy) / L2
        if t < 0:   t = 0
        if t > 1:   t = 1

        nx, ny = (x1 + t * vx), (y1 + t * vy)
        dist = ((px - nx) ** 2 + (py - ny) ** 2) ** 0.5
        if dist < best_dist:
            # heading langs segment (E->N = x->y)
            hd = (math.degrees(math.atan2(vx, vy)) + 360.0) % 360.0
            best_hd, best_dist, best_proj = hd, dist, (nx, ny)

    return best_hd, best_dist, best_proj

def heading_from_lines(E, N):
    """
    Slår opp nærmeste linjesegment, returnerer (heading_deg, avstand).
    Reprojiserer linjer til punkt-CRS ved behov og ignorerer tomme features.
    """
    lines = st.session_state.get("LINES_LIST") or []
    if not lines:
        return (None, None)

    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    epsg_lin = st.session_state.get("LINES_EPSG", 25832)
    buf = st.session_state.get("BUFFER_M", 2.0)

    # reprojiser linjer til samme CRS som punktene
    if epsg_lin != epsg_pts:
        tr = Transformer.from_crs(epsg_lin, epsg_pts, always_xy=True)
        def reproj(coords):
            return [tuple(tr.transform(x, y)) for (x, y) in coords]
        use = [{"coords": reproj(L.get("coords", [])), "objtype": L.get("objtype")}
               for L in lines if isinstance(L, dict) and L.get("coords")]
    else:
        use = [L for L in lines if isinstance(L, dict) and L.get("coords")]

    if not use:
        return (None, None)

    best_hd, best_dist = None, float("inf")
    for L in use:
        hd, dist, _ = nearest_heading_on_polyline(L["coords"], (E, N))
        if dist is not None and dist < best_dist:
            best_hd, best_dist = hd, dist

    if best_dist == float("inf"):
        return (None, None)

    return (best_hd if best_dist <= buf else None), best_dist


def choose_pos_and_heading(sobj_label=None, E=None, N=None, Alt=None, Rot=None, manual_override=None):
    centers = st.session_state.get("CENTERS_DICT") or {}
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    center_hint = None
    if (E is None or N is None) and sobj_label:
        base = base_id(sobj_label)
        info = centers.get(base)
        if info:
            E = info.get("center_E")
            N = info.get("center_N")
            if Alt is None:
                Alt = info.get("hoyde")
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

# ------------------------- Tabs -------------------------

tabA, tabB, tabC, tabD = st.tabs([
    "A) Batch geotagg",
    "B) Målebok (punkter+bilder)",
    "C) Manuell / kart / 2-klikk",
    "D) Kart"
])

# ------------------------- Tab A: Batch -------------------------

with tabA:
    st.subheader("A) Geotagg mange bilder")
    centers_dict = st.session_state.get("CENTERS_DICT") or {}
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    draw_arrow = st.session_state.get("DRAW_ARROW", True)
    arrow_size = st.session_state.get("ARROW_SIZE", 120)
    n_label_size = st.session_state.get("N_LABEL_SIZE", 18)
    arrow_col = st.session_state.get("ARROW_COLOR", (255, 255, 255))
    arrow_outline = st.session_state.get("ARROW_OUTLINE", (0, 0, 0))

    picked_label = None
    if centers_dict:
        options = sorted(list(centers_dict.keys()))
        picked_label = st.selectbox("Velg kum/S_OBJID", options, key="A_pick_label")
    else:
        st.warning("Ingen kum-senter i prosjektdata ennå. Last opp punkter i sidepanelet.")

    mode = st.radio("Bildekilde:", ["ZIP-opplasting", "Opplasting (flere filer)"], index=1, key="A_mode")
    exts_ok = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif")
    patt = st.selectbox("Nytt filnavn", ["keep", "label_orig", "label_only", "label_en"], index=1, key="A_rename")
    epsg_out = ensure_epsg("TAB_A_DOC_EPSG", "Dokumentasjons-CRS for CSV (eksport)", default=25832)

    with st.expander("Avansert: manuell heading/Nord (grader)"):
        man_enable = st.checkbox("Overstyr heading manuelt", value=False, key="A_manual_on")
        man_heading = st.number_input("Manuell heading (0–359°)", min_value=0, max_value=359, value=0, key="A_manual_val")

    zip_up = None
    files_up = None
    if mode == "ZIP-opplasting":
        zip_up = st.file_uploader("Last opp ZIP med bilder", type=["zip"], key="A_zip2")
    else:
        files_up = st.file_uploader("Dra inn flere bilder", type=[e[1:] for e in exts_ok], accept_multiple_files=True, key="A_files2")
        if not HEIC_OK:
            st.info("HEIC/HEIF krever 'pillow-heif'.")

    def load_any_to_jpeg_bytes(payload: bytes) -> bytes:
        im = Image.open(io.BytesIO(payload))
        im = normalize_orientation(im).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "jpeg", quality=95)
        return buf.getvalue()

    def build_new_name(pattern, label, orig_name, E=None, N=None):
        base, _ext = os.path.splitext(orig_name)
        safe = re.sub(r'[\\/:*?"<>|]+', "_", str(label)).strip().replace(" ", "_")
        if pattern == "keep" or not safe:
            return f"{base}.jpg"
        if pattern == "label_orig":
            return f"{safe}_{base}.jpg"
        if pattern == "label_only":
            return f"{safe}.jpg"
        if pattern == "label_en":
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
                E0 = info.get("center_E")
                N0 = info.get("center_N")
                Alt0 = info.get("hoyde")
                if E0 is None or N0 is None:
                    st.error(f"Kum '{picked_label}' mangler E/N.")
                else:
                    processed = []
                    skipped = []
                    zout_mem = io.BytesIO()
                    zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)
                    used = set()

                    def process_one(name, payload):
                        name = str(name)
                        if not name.lower().endswith(exts_ok):
                            skipped.append({"file": name, "reason": "Ikke-støttet filtype"})
                            return
                        try:
                            jpeg0 = load_any_to_jpeg_bytes(payload)
                        except Exception as e:
                            skipped.append({"file": name, "reason": f"Kunne ikke lese bilde: {e}"})
                            return

                        E, N, Alt, hd, cent, line_h, dist = choose_pos_and_heading(
                            picked_label, E0, N0, Alt0, None,
                            manual_override=(float(man_heading) if man_enable else None)
                        )
                        if E is None or N is None:
                            skipped.append({"file": name, "reason": "Mangler E/N"})
                            return

                        use_hd = hd if _is_valid_number(hd) else (float(man_heading) if man_enable else None)
                        use_hd = apply_heading_calibration(use_hd)

                        lat, lon = transform_EN_to_wgs84(E, N, epsg_pts)
                        im = Image.open(io.BytesIO(jpeg0))
                        if st.session_state.get("DRAW_ARROW", True) and _is_valid_number(use_hd):
                            im = draw_north_arrow(
                                im,
                                _wrap_deg(use_hd),
                                size_px=st.session_state.get("ARROW_SIZE", 120),
                                color=st.session_state.get("ARROW_COLOR", (255, 255, 255)),
                                outline=st.session_state.get("ARROW_OUTLINE", (0, 0, 0)),
                                n_label_size=st.session_state.get("N_LABEL_SIZE", 18),
                            )
                        buf = io.BytesIO()
                        im.save(buf, "jpeg", quality=95)
                        buf.seek(0)
                        jpeg1 = write_exif_jpeg_bytes(buf.getvalue(), lat, lon, Alt, use_hd)

                        Xdoc, Ydoc = transform_EN_to_epsg(E, N, epsg_pts, epsg_out)

                        if _is_valid_number(line_h):
                            heading_source = "linje"
                        elif _is_valid_number(cent):
                            heading_source = "kum-azimut"
                        elif _is_valid_number(use_hd):
                            heading_source = "manuell/kalibrert"
                        else:
                            heading_source = "ukjent"

                        processed.append(
                            {
                                "file": name,
                                "S_OBJID": picked_label,
                                "E_in": E,
                                "N_in": N,
                                "lat": lat,
                                "lon": lon,
                                f"E_{epsg_out}": Xdoc,
                                f"N_{epsg_out}": Ydoc,
                                "hoyde": Alt,
                                "heading": use_hd,
                                "heading_line": line_h,
                                "center_hint": cent,
                                "dist_to_line": dist,
                                "heading_source": heading_source,
                            }
                        )

                        newname = build_new_name(patt, picked_label, os.path.basename(name), E0, N0)
                        base, ext = os.path.splitext(newname)
                        cand = newname
                        i = 1
                        while cand in used:
                            cand = f"{base}_{i}.jpg"
                            i += 1
                        used.add(cand)
                        zout.writestr(cand, jpeg1)

                    if mode == "ZIP-opplasting":
                        if not zip_up:
                            st.error("Last opp ZIP.")
                        else:
                            zin = zipfile.ZipFile(io.BytesIO(zip_up.read()), "r")
                            for name in zin.namelist():
                                if not name.lower().endswith(exts_ok):
                                    continue
                                process_one(os.path.basename(name), zin.read(name))
                    else:
                        if not files_up or len(files_up) == 0:
                            st.error("Dra inn bilder.")
                        else:
                            for f in files_up:
                                process_one(f.name, f.read())

                    zout.close()

                    if processed:
                        st.download_button("Last ned ZIP", data=zout_mem.getvalue(), file_name="geotagged.zip", mime="application/zip")
                        st.success(f"Geotagget {len(processed)} bilder.")
                        dfo = pd.DataFrame(processed)
                        st.markdown("**Debug: første 10 rader (heading-kilde, dist_to_line m.m.)**")
                        st.dataframe(dfo.head(10), use_container_width=True)
                        st.download_button("Last ned CSV (Tab A)", dfo.to_csv(index=False).encode("utf-8"), "geotag_tabA.csv", "text/csv")
                    else:
                        st.error("Ingen bilder skrevet (0).")

                    if skipped:
                        st.warning("Hoppet over:")
                        st.dataframe(pd.DataFrame(skipped))
        except Exception as e:
            st.exception(e)

# ------------------------- Tab B: Målebok -------------------------

with tabB:
    st.subheader("B) Målebok – punkter + bilder")
    st.write("Last opp en **punkttabell** (Excel/CSV) fra måleboka med minst kolonnene: Øst, Nord, (Høyde), (Rotasjon), S_OBJID og S_HYPERLINK (bildefilnavn). Last opp så bildene (ZIP eller mappe).")
    pts_file = st.file_uploader("Punkter (Excel/CSV)", type=["xlsx", "xls", "csv"], key="B_pts")
    epsg_pts_B = ensure_epsg("B_epsg_pts", "EPSG for punkter (målebok)", default=st.session_state.get("POINTS_EPSG", 25832))
    img_mode_B = st.radio("Bildekilde:", ["ZIP-opplasting", "Opplasting (flere filer)"], index=0, key="B_mode")
    zip_up_B = None
    files_up_B = None
    if img_mode_B == "ZIP-opplasting":
        zip_up_B = st.file_uploader("Last opp ZIP med bilder", type=["zip"], key="B_zip")
    else:
        files_up_B = st.file_uploader("Dra inn flere bilder", type=["jpg", "jpeg", "png", "tif", "tiff", "heic", "heif"], accept_multiple_files=True, key="B_files")

    if st.button("Kjør (Tab B)"):
        try:
            if pts_file is None:
                st.error("Manglende punkttabell.")
            else:
                df = dataframe_from_upload(pts_file)
                cols = detect_columns(df)
                miss = [k for k in ["east", "north", "sobj", "hyper"] if cols.get(k) is None]
                if miss:
                    st.error(f"Mangler kolonner i tabellen: {', '.join(miss)}")
                else:
                    # last bildebiter
                    payloads = {}
                    if zip_up_B is not None:
                        zin = zipfile.ZipFile(io.BytesIO(zip_up_B.read()), "r")
                        for name in zin.namelist():
                            base = os.path.basename(name)
                            if base:
                                payloads[base.lower()] = zin.read(name)
                    elif files_up_B:
                        for f in files_up_B:
                            payloads[os.path.basename(f.name).lower()] = f.read()
                    else:
                        st.error("Ingen bilder lastet.")
                        raise Exception("No images")

                    arrow_size = st.session_state.get("ARROW_SIZE", 120)
                    n_label_size = st.session_state.get("N_LABEL_SIZE", 18)
                    arrow_col = st.session_state.get("ARROW_COLOR", (255, 255, 255))
                    arrow_outline = st.session_state.get("ARROW_OUTLINE", (0, 0, 0))

                    out_csv = []
                    zout_mem = io.BytesIO()
                    zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)
                    count = 0
                    skipped = []

                    for _, r in df.iterrows():
                        sobj = str(r[cols["sobj"]]).strip()
                        imgname = os.path.basename(str(r[cols["hyper"]])).strip()
                        if not imgname:
                            skipped.append({"row": int(_), "reason": "Tomt bildefilnavn"})
                            continue
                        payload = payloads.get(imgname.lower())
                        if payload is None:
                            skipped.append({"row": int(_), "img": imgname, "reason": "Bilde ikke funnet"})
                            continue
                        E = parse_float_maybe_comma(r[cols["east"]])
                        N = parse_float_maybe_comma(r[cols["north"]])
                        Alt = parse_float_maybe_comma(r[cols["alt"]]) if cols["alt"] else None
                        Rot = parse_float_maybe_comma(r[cols["rot"]]) if cols["rot"] else None
                        if E is None or N is None:
                            skipped.append({"row": int(_), "img": imgname, "reason": "Manglende E/N"})
                            continue

                        # heading-kandidater
                        line_h, _dist = heading_from_lines(E, N)
                        hd = line_h if _is_valid_number(line_h) else (Rot if _is_valid_number(Rot) else None)
                        hd = apply_heading_calibration(hd)

                        lat, lon = transform_EN_to_wgs84(E, N, epsg_pts_B)
                        im = Image.open(io.BytesIO(payload))
                        im = normalize_orientation(im).convert("RGB")
                        if st.session_state.get("DRAW_ARROW", True) and _is_valid_number(hd):
                            im = draw_north_arrow(im, _wrap_deg(hd), size_px=arrow_size, color=arrow_col, outline=arrow_outline, n_label_size=n_label_size)
                        buf = io.BytesIO()
                        im.save(buf, "jpeg", quality=95)
                        buf.seek(0)
                        jpeg1 = write_exif_jpeg_bytes(buf.getvalue(), lat, lon, Alt, hd)
                        outname = f"{base_id(sobj)}_{os.path.splitext(imgname)[0]}.jpg"
                        zout.writestr(outname, jpeg1)
                        count += 1

                        out_csv.append({"S_OBJID": sobj, "file_in": imgname, "file_out": outname, "E": E, "N": N, "hoyde": Alt, "heading": hd})

                    zout.close()
                    if count > 0:
                        st.download_button("Last ned ZIP (Tab B)", data=zout_mem.getvalue(), file_name="geotag_malebok.zip", mime="application/zip")
                        st.download_button("Last ned CSV (Tab B)", data=pd.DataFrame(out_csv).to_csv(index=False).encode("utf-8"), file_name="tabB_result.csv", mime="text/csv")
                        st.success(f"Skrev {count} bilder.")
                    if skipped:
                        st.warning("Hoppet over:")
                        st.dataframe(pd.DataFrame(skipped))
        except Exception as e:
            st.exception(e)

# ------------------------- Tab C: Manuell/kart/2-klikk -------------------------

with tabC:
    st.subheader("C) Manuell / kart / 2-klikk")

    centers_dict = st.session_state.get("CENTERS_DICT") or {}
    epsg_pts = st.session_state.get("POINTS_EPSG", 25832)
    draw_arrow = st.session_state.get("DRAW_ARROW", True)
    arrow_size = st.session_state.get("ARROW_SIZE", 120)
    n_label_size = st.session_state.get("N_LABEL_SIZE", 18)
    arrow_col = st.session_state.get("ARROW_COLOR", (255, 255, 255))
    arrow_outline = st.session_state.get("ARROW_OUTLINE", (0, 0, 0))

      # Enkle kartinnstillinger
    line_width_px   = st.slider("Linjebredde (px)", 0.1, 12, 0.8, 0.1)
    center_size_px  = st.slider("Punktstørrelse – kum-senter (px)", 0.1, 30, 3, 0.1)
    corner_size_px  = st.slider("Punktstørrelse – hjørne (px)", 0.1, 12, 1.0, 0.1)
    show_center_lbl = st.checkbox("Etikett på kum-senter (base_id)", value=True)
    show_corner_lbl = st.checkbox("Etikett på hjørner (idx)", value=False)

    m = folium.Map(
    location=[lat0, lon0],
    zoom_start=19,
    tiles=None,              # vi legger inn egne lag under
    control_scale=True,
    prefer_canvas=True,      # <— viktig for sub-1 px
    max_zoom=23              # <— la kartet få gå dypere
)

    # OSM (vanlig, maks ~19)
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="© OpenStreetMap contributors",
    name="OSM",
    max_zoom=19,
    overlay=False,
    control=True,
).add_to(m)

# Esri World Imagery – høy maks zoom (ofte 22–23)
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Esri imagery",
    max_zoom=23,
    overlay=False,
    control=True,
).add_to(m)

# Valgfritt: Esri World Street Map
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri streets",
    max_zoom=23,
    overlay=False,
    control=True,
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)


    if centers_dict:
        options = sorted(list(centers_dict.keys()))
        picked_label_C = st.selectbox("Velg kum/S_OBJID", options, key="C_pick_label")
    else:
        picked_label_C = None
        st.warning("Last opp punkter i sidepanelet.")

    files_up_C = st.file_uploader("Dra inn bilder (flere)", type=["jpg", "jpeg", "png", "tif", "tiff", "heic", "heif"], accept_multiple_files=True, key="C_files")
    if files_up_C and len(files_up_C) > 0 and picked_label_C:
        if "MANUAL_HEADINGS" not in st.session_state:
            st.session_state["MANUAL_HEADINGS"] = {}
        man_dict = st.session_state["MANUAL_HEADINGS"]
        names = [f.name for f in files_up_C]
        sel = st.selectbox("Velg bilde", names, key="C_sel_name")
        cur_idx = names.index(sel) if sel in names else 0

        colL, colR = st.columns([2, 1])

        with colL:
            f = files_up_C[cur_idx]
            payload = f.read()
            f.seek(0)
            im0 = Image.open(io.BytesIO(payload))
            im0 = normalize_orientation(im0).convert("RGB")
            info = centers_dict.get(picked_label_C, {})
            E0 = info.get("center_E")
            N0 = info.get("center_N")
            Alt0 = info.get("hoyde")

            E, N, Alt, hd, cent, line_h, dist = choose_pos_and_heading(picked_label_C, E0, N0, Alt0, None, manual_override=None)
            cur_manual = man_dict.get(sel)
            show_hd = cur_manual if _is_valid_number(cur_manual) else hd
            show_hd = apply_heading_calibration(show_hd)

            if draw_arrow and _is_valid_number(show_hd):
                im_prev = draw_north_arrow(im0.copy(), _wrap_deg(show_hd), size_px=arrow_size, color=arrow_col, outline=arrow_outline, n_label_size=n_label_size)
            else:
                im_prev = im0

            st.image(im_prev, caption=f"Forhåndsvisning – heading={show_hd if show_hd is not None else '—'}°", use_column_width=True)

            with st.expander("Orienter i KART (klikk eller tegn/drag linje)"):
                if E0 is not None and N0 is not None:
                    lat0, lon0 = transform_EN_to_wgs84(E0, N0, epsg_pts)
                    base_hd = man_dict.get(sel, hd if _is_valid_number(hd) else 0.0) or 0.0
                    base_hd = float(base_hd)
                    Lm = st.slider("Linjelengde (meter)", 2.0, 20.0, 8.0, 0.5, key="C_line_len")
                    adj = st.slider("Finjustering (°)", -180, 180, 0, 1, key="C_line_adj")

                    m = folium.Map(location=[lat0, lon0], zoom_start=19, tiles="OpenStreetMap", control_scale=True)
                    folium.CircleMarker([lat0, lon0], radius=5, color="#0096ff", fill=True, fill_opacity=0.9, tooltip=f"{picked_label_C}").add_to(m)

                    # Tegn VA/EL-linjer i Tab C-kartet
                    lines_list = st.session_state.get("LINES_LIST") or []
                    epsg_lin = st.session_state.get("LINES_EPSG", 25832)

                    def to_wgs_list_C(coords, src_epsg):
                        if src_epsg == 4326:
                            return [(y, x) for (x, y) in [(c[0], c[1]) for c in coords]]
                        trL = Transformer.from_crs(src_epsg, 4326, always_xy=True)
                        out = []
                        for (x, y) in coords:
                            lon, lat = trL.transform(x, y)
                            out.append((lat, lon))
                        return out

                    if lines_list:
                        fg_lines_C = folium.FeatureGroup(name="Linjer (VA/EL)").add_to(m)
                        for L in lines_list:
                            path_latlon = to_wgs_list_C(L["coords"], epsg_lin)
                            folium.PolyLine(path_latlon, color="#5050C8", weight=5, opacity=0.9,
                                            tooltip=L.get("objtype") or "linje").add_to(fg_lines_C)


                    # Hjornemarkører (fra opplastet punkttabell)
                    pts_df = st.session_state.get("POINTS_DF")
                    delims_val = st.session_state.get("SB_delims", "-_ ./")
                    if pts_df is not None:
                        cols_all = detect_columns(pts_df)
                        if cols_all["sobj"] and cols_all["east"] and cols_all["north"]:
                            tmp = pts_df.copy()
                            tmp["_base"] = tmp[cols_all["sobj"]].astype(str).map(lambda s: base_id(s, delims_val))
                            grp = tmp[tmp["_base"] == base_id(picked_label_C, delims_val)].reset_index(drop=True)
                            for i, r in grp.iterrows():
                                e = parse_float_maybe_comma(r[cols_all["east"]])
                                n = parse_float_maybe_comma(r[cols_all["north"]])
                                if _is_valid_number(e) and _is_valid_number(n):
                                    lt, ln = transform_EN_to_wgs84(e, n, epsg_pts)
                                    folium.CircleMarker([lt, ln], radius=3, color="#00cc00", fill=True, fill_opacity=0.9, tooltip=f"hj[{i}]").add_to(m)

                    # Forhåndslinje fra heading
                    def latlon_from_heading(Ec, Nc, hd_deg, length_m):
                        rad = math.radians(hd_deg)
                        E1 = Ec + math.sin(rad) * length_m
                        N1 = Nc + math.cos(rad) * length_m
                        lt, ln = transform_EN_to_wgs84(E1, N1, epsg_pts)
                        return lt, ln

                    lt1, ln1 = latlon_from_heading(E0, N0, base_hd + adj, Lm)
                    folium.PolyLine(locations=[[lat0, lon0], [lt1, ln1]], color="#c83c3c", weight=5, tooltip=f"{base_hd+adj:.1f}°").add_to(m)
                    folium.Marker([lt1, ln1], draggable=False, icon=folium.Icon(color="red", icon="compass")).add_to(m)

                    # Draw-plugin: polyline/marker for å angi retning
                    draw = Draw(
                        export=True,
                        filename='heading.json',
                        position='topleft',
                        draw_options={'polyline': True, 'polygon': False, 'rectangle': False, 'circle': False, 'marker': True, 'circlemarker': False},
                        edit_options={'edit': True, 'remove': True}
                    )
                    draw.add_to(m)
                    folium.LatLngPopup().add_to(m)
                    out = st_folium(m, height=440, width=None)

                    new_hd = None
                    # 1) Klikk i kartet
                    if out and out.get("last_clicked") is not None:
                        latc, lonc = out["last_clicked"]["lat"], out["last_clicked"]["lng"]
                        tr = Transformer.from_crs(4326, epsg_pts, always_xy=True)
                        Ex, Ny = tr.transform(lonc, latc)
                        dx = Ex - E0
                        dy = Ny - N0
                        new_hd = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

                    # 2) Tegnet geometri (Polyline/Marker)
                    if out and out.get("last_active_drawing"):
                        d = out["last_active_drawing"]
                        typ = d.get("type")
                        if typ == "polyline":
                            coords = d["geometry"]["coordinates"]
                            if len(coords) >= 2:
                                lonc, latc = coords[-1]
                                tr = Transformer.from_crs(4326, epsg_pts, always_xy=True)
                                Ex, Ny = tr.transform(lonc, latc)
                                dx = Ex - E0
                                dy = Ny - N0
                                new_hd = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
                        elif typ == "marker":
                            lonc, latc = d["geometry"]["coordinates"]
                            tr = Transformer.from_crs(4326, epsg_pts, always_xy=True)
                            Ex, Ny = tr.transform(lonc, latc)
                            dx = Ex - E0
                            dy = Ny - N0
                            new_hd = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

                    if _is_valid_number(new_hd):
                        st.info(f"Ny heading fra kart: {new_hd:.1f}°")

                    if st.button("Lagre heading (kart)", key="C_save_map"):
                        final_hd = new_hd if _is_valid_number(new_hd) else (base_hd + adj)
                        st.session_state["MANUAL_HEADINGS"][sel] = float(_wrap_deg(final_hd))
                        st.success(f"Lagret {sel}: {_wrap_deg(final_hd):.1f}°")
                else:
                    st.warning("Mangler E/N for kum-senter.")

            with st.expander("Orienter med 2 hjørner (klikk i bildet)"):
                pts_df = st.session_state.get("POINTS_DF")
                delims_val = st.session_state.get("SB_delims", "-_ ./")
                base_lbl = base_id(picked_label_C, delims_val) if picked_label_C else None
                corners_df = None
                if pts_df is not None and base_lbl:
                    cols = detect_columns(pts_df)
                    if cols["sobj"] and cols["east"] and cols["north"]:
                        tmp = pts_df.copy()
                        tmp["_base"] = tmp[cols["sobj"]].astype(str).map(lambda s: base_id(s, delims_val))
                        grp = tmp[tmp["_base"] == base_lbl]
                        if len(grp) >= 2:
                            corners_df = grp.rename(columns={cols["east"]: "E", cols["north"]: "N"}).reset_index(drop=True)

                if corners_df is None or len(corners_df) < 2:
                    st.info("Finner ikke minst 2 hjørner for denne kummen i opplastet punktfil.")
                else:
                    corners_df = corners_df.reset_index(drop=True).copy()
                    corners_df["idx"] = corners_df.index
                    show_cols = ["idx"]
                    if "S_OBJID" in corners_df.columns:
                        show_cols.append("S_OBJID")
                    show_cols += [c for c in ["E", "N"] if c in corners_df.columns]
                    st.dataframe(corners_df[show_cols].head(30), use_container_width=True)

                    st.caption("Klikk først Hjørne A, deretter Hjørne B i bildet:")
                    click_key = f"C_clicks_{picked_label_C}_{sel}"
                    coords = img_coords(im0, key=click_key)
                    if coords:
                        clicks = st.session_state.get(click_key + "_list", [])
                        if (not clicks) or (clicks and (clicks[-1] != (coords["x"], coords["y"]))):
                            clicks.append((coords["x"], coords["y"]))
                            clicks = clicks[-2:]
                            st.session_state[click_key + "_list"] = clicks
                    clicks = st.session_state.get(click_key + "_list", [])
                    st.write(f"Klikk: {clicks}")

                    idxA = st.number_input("Indeks hjørne A (radnr 0..)", min_value=0, max_value=len(corners_df) - 1, value=0, key="C_cA_idx")
                    idxB = st.number_input("Indeks hjørne B (radnr 0..)", min_value=0, max_value=len(corners_df) - 1, value=min(1, len(corners_df) - 1), key="C_cB_idx")
                    EA = parse_float_maybe_comma(corners_df.loc[idxA, "E"]) if 0 <= idxA < len(corners_df) else None
                    NA = parse_float_maybe_comma(corners_df.loc[idxA, "N"]) if 0 <= idxA < len(corners_df) else None
                    EB = parse_float_maybe_comma(corners_df.loc[idxB, "E"]) if 0 <= idxB < len(corners_df) else None
                    NB = parse_float_maybe_comma(corners_df.loc[idxB, "N"]) if 0 <= idxB < len(corners_df) else None

                    if _is_valid_number(EA) and _is_valid_number(NA) and _is_valid_number(EB) and _is_valid_number(NB):
                        if len(clicks) == 2:
                            (xA, yA), (xB, yB) = clicks
                            # virkelig azimut mellom hjørner (E/N i meter)
                            az_real = (math.degrees(math.atan2(EB - EA, NB - NA)) + 360.0) % 360.0
                            # retning i bildekoordinater (x høyre, y ned)
                            az_img = (math.degrees(math.atan2(xB - xA, -(yB - yA))) + 360.0) % 360.0
                            hd2 = (az_real - az_img) % 360.0
                            st.success(f"Beregnet heading = {hd2:.1f}° (lagres som manuell)")
                            st.session_state["MANUAL_HEADINGS"][sel] = float(hd2)
                            im_prev2 = draw_north_arrow(im0.copy(), _wrap_deg(hd2), size_px=arrow_size, color=arrow_col, outline=arrow_outline, n_label_size=n_label_size)
                            st.image(im_prev2, caption=f"Forhåndsvisning (2-klikk) = {hd2:.1f}°", use_column_width=True)
                        if st.button("Nullstill 2-klikk", key="C_clearclicks_btn"):
                            st.session_state[click_key + "_list"] = []
                    else:
                        st.warning("Ugyldige E/N for valgte hjørner.")

        with colR:
            st.markdown("**Sett heading for valgt bilde**")
            if not _is_valid_number(hd):
                st.info("Ingen heading funnet automatisk – sett manuelt, i kart eller med 2-klikk.")
            cur_manual = st.session_state["MANUAL_HEADINGS"].get(sel)
            base_val = int(cur_manual if _is_valid_number(cur_manual) else int(hd or 0))
            man_val = st.slider("Manuell heading (0–359°)", 0, 359, base_val, key="C_slider")
            c1, c2, c3 = st.columns(3)
            if c1.button("−10°", key="C_m10"):
                man_val = (man_val - 10) % 360
                st.session_state["C_slider"] = man_val
            if c2.button("+10°", key="C_p10"):
                man_val = (man_val + 10) % 360
                st.session_state["C_slider"] = man_val
            if c3.button("Flip 180°", key="C_flip"):
                man_val = (man_val + 180) % 360
                st.session_state["C_slider"] = man_val
            if st.button("Lagre heading (manuell)", key="C_save_one"):
                st.session_state["MANUAL_HEADINGS"][sel] = float(man_val)
                st.success(f"Lagret {sel}: {man_val}°")

        st.markdown("---")
        if st.button("Eksporter alle som ZIP (med heading der satt)", key="C_export"):
            processed = 0
            skipped = []
            zout_mem = io.BytesIO()
            zout = zipfile.ZipFile(zout_mem, "w", zipfile.ZIP_DEFLATED)
            for f in files_up_C:
                try:
                    payload = f.read()
                    f.seek(0)
                    im0 = Image.open(io.BytesIO(payload))
                    im0 = normalize_orientation(im0).convert("RGB")
                    info = centers_dict.get(picked_label_C, {})
                    E0 = info.get("center_E")
                    N0 = info.get("center_N")
                    Alt0 = info.get("hoyde")

                    man = st.session_state["MANUAL_HEADINGS"].get(f.name)
                    if man is None and f.name == sel:
                        man = st.session_state["MANUAL_HEADINGS"].get(sel)

                    E, N, Alt, hd, cent, line_h, dist = choose_pos_and_heading(picked_label_C, E0, N0, Alt0, None, manual_override=man if man is not None else None)
                    if E is None or N is None:
                        skipped.append({"file": f.name, "reason": "Mangler E/N"})
                        continue

                    hd = apply_heading_calibration(hd)

                    lat, lon = transform_EN_to_wgs84(E, N, epsg_pts)
                    if draw_arrow and _is_valid_number(hd):
                        im0 = draw_north_arrow(im0, _wrap_deg(hd), size_px=arrow_size, color=arrow_col, outline=arrow_outline, n_label_size=n_label_size)
                    buf = io.BytesIO()
                    im0.save(buf, "jpeg", quality=95)
                    buf.seek(0)
                    jpeg1 = write_exif_jpeg_bytes(buf.getvalue(), lat, lon, Alt, hd)
                    newname = f"{picked_label_C}_{os.path.splitext(os.path.basename(f.name))[0]}.jpg"
                    zout.writestr(newname, jpeg1)
                    processed += 1
                except Exception as e:
                    skipped.append({"file": f.name, "reason": str(e)})
            zout.close()
            if processed > 0:
                st.download_button("Last ned ZIP (Tab C)", data=zout_mem.getvalue(), file_name="geotag_manual.zip", mime="application/zip")
                st.success(f"Skrev {processed} bilder.")
            if skipped:
                st.warning("Hoppet over:")
                st.dataframe(pd.DataFrame(skipped))
    else:
        st.info("Last opp bilder og velg kum for manuell forhåndsvisning.")

# ------------------------- Tab D: Oversiktskart -------------------------

with tabD:
    st.subheader("D) Kart – oversikt (Folium)")

    # Bruk data fra prosjekt/minne
    lines       = st.session_state.get("LINES_LIST") or []
    centers_df  = st.session_state.get("CENTERS_DF")
    pts_df      = st.session_state.get("POINTS_DF")
    epsg_pts    = st.session_state.get("POINTS_EPSG", 25832)
    epsg_lin    = st.session_state.get("LINES_EPSG", 25832)

    # Enkle kartinnstillinger
    line_width_px   = st.slider("Linjebredde (px)", 0.1, 12, 0.8, 0.1)
    center_size_px  = st.slider("Punktstørrelse – kum-senter (px)", 0.1, 30, 3, 0.1)
    corner_size_px  = st.slider("Punktstørrelse – hjørne (px)", 0.1, 12, 1.0, 0.1)
    show_center_lbl = st.checkbox("Etikett på kum-senter (base_id)", value=True)
    show_corner_lbl = st.checkbox("Etikett på hjørner (idx)", value=False)

    # Velg kartets sentrum
    lat0, lon0 = 59.91, 10.75  # fallback Oslo
    try:
        if centers_df is not None and not centers_df.empty:
            trc = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
            lons, lats = [], []
            for e, n in zip(centers_df["center_E"], centers_df["center_N"]):
                x, y = trc.transform(float(e), float(n))
                lons.append(x); lats.append(y)
            lat0, lon0 = float(pd.Series(lats).mean()), float(pd.Series(lons).mean())
        elif lines:
            # Bruk første koordinat i linjene
            if epsg_lin != 4326:
                trl = Transformer.from_crs(epsg_lin, 4326, always_xy=True)
                x, y = trl.transform(lines[0]["coords"][0][0], lines[0]["coords"][0][1])
            else:
                x, y = lines[0]["coords"][0]
            lat0, lon0 = y, x  # (lat, lon)
    except Exception:
        pass

    # Start folium-kart
    
    m = folium.Map(
    location=[lat0, lon0],
    zoom_start=19,
    tiles=None,              # vi legger inn egne lag under
    control_scale=True,
    prefer_canvas=True,      # <— viktig for sub-1 px
    max_zoom=23              # <— la kartet få gå dypere
)

    # OSM (vanlig, maks ~19)
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="© OpenStreetMap contributors",
    name="OSM",
    max_zoom=19,
    overlay=False,
    control=True,
).add_to(m)

# Esri World Imagery – høy maks zoom (ofte 22–23)
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    name="Esri imagery",
    max_zoom=23,
    overlay=False,
    control=True,
).add_to(m)

# Valgfritt: Esri World Street Map
folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri streets",
    max_zoom=23,
    overlay=False,
    control=True,
).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)


    # ---------- Linjer (VA/EL) ----------
    fg_lines = folium.FeatureGroup(name="Linjer (VA/EL)").add_to(m)

    def to_wgs_list(coords, src_epsg):
        if src_epsg == 4326:
            # coords = [(lon, lat)]
            return [(y, x) for (x, y) in [(c[0], c[1]) for c in coords]]  # sikre (lat, lon)
        tr = Transformer.from_crs(src_epsg, 4326, always_xy=True)
        out = []
        for (x, y) in coords:
            lon, lat = tr.transform(x, y)
            out.append((lat, lon))
        return out

    if lines:
        try:
            for L in lines:
                path_latlon = to_wgs_list(L["coords"], epsg_lin)
                folium.PolyLine(
                    locations=path_latlon,
                    color="#5050C8",
                    weight=line_width_px,
                    opacity=0.9,
                    tooltip=L.get("objtype") or "linje",
                ).add_to(fg_lines)
        except Exception as e:
            st.warning(f"Kunne ikke tegne linjer: {e}")
    else:
        st.info("Ingen linjer lastet eller tolket fra filen.")

    # ---------- Kum-sentre ----------
    fg_centers = folium.FeatureGroup(name="Kum-senter").add_to(m)
    if centers_df is not None and not centers_df.empty:
        try:
            trc = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
            for _, r in centers_df.iterrows():
                lon, lat = trc.transform(float(r["center_E"]), float(r["center_N"]))
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=center_size_px,
                    color="#0096ff",
                    fill=True, fill_opacity=0.9,
                    tooltip=str(r["base_id"]) if show_center_lbl else None,
                ).add_to(fg_centers)
        except Exception as e:
            st.warning(f"Kunne ikke tegne kum-sentre: {e}")

    # ---------- Hjørnepunkter ----------
    fg_corners = folium.FeatureGroup(name="Hjørnepunkter").add_to(m)
    if pts_df is not None:
        cols = detect_columns(pts_df)
        if cols["east"] and cols["north"]:
            try:
                trp = Transformer.from_crs(epsg_pts, 4326, always_xy=True)
                # merk idx for hvert _base (grunn-ID)
                delims_val = st.session_state.get("SB_delims", "-_ ./")
                tmp = pts_df.copy()
                if cols["sobj"]:
                    tmp["_base"] = tmp[cols["sobj"]].astype(str).map(lambda s: base_id(s, delims_val))
                else:
                    tmp["_base"] = ""

                tmp = tmp.reset_index(drop=True)
                tmp["idx"] = tmp.groupby("_base").cumcount()

                for _, rr in tmp.iterrows():
                    e = parse_float_maybe_comma(rr[cols["east"]])
                    n = parse_float_maybe_comma(rr[cols["north"]])
                    if not _is_valid_number(e) or not _is_valid_number(n):
                        continue
                    lon, lat = trp.transform(float(e), float(n))
                    tip = f'{rr.get("_base","")}[{rr["idx"]}]' if show_corner_lbl else None
                    folium.CircleMarker(
                        location=(lat, lon),
                        radius=corner_size_px,
                        color="#00cc00",
                        fill=True, fill_opacity=0.9,
                        tooltip=tip,
                    ).add_to(fg_corners)
            except Exception as e:
                st.warning(f"Kunne ikke tegne hjørner: {e}")

    # Slå av/på lag i kartet
    folium.LayerControl(collapsed=False).add_to(m)

    # Render kartet
    st_folium(m, height=600, width=None)


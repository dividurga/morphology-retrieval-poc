"""uniformly rescale walker2d_v5.xml's geometry. lengths, radii, joint positions and the root
height reference scale by s, motor gear by s^4 (torque ~ density*L^4). density, damping and armature
are left alone.

    uv run python walker/scale_mjcf.py 0.532 > walker/walker2d_scaled.xml
"""

import sys
import xml.etree.ElementTree as ET


def scale(path: str, s: float) -> str:
    tree = ET.parse(path)
    for el in tree.getroot().iter():
        if el.tag == "geom" and el.get("type") == "plane":
            continue  # the floor stays infinite
        for attr in ("pos", "fromto", "size"):
            if el.tag in ("camera", "light") or el.get(attr) is None:
                continue
            el.set(attr, " ".join(f"{float(v) * s:.8g}" for v in el.get(attr).split()))
        if el.tag == "motor" and el.get("gear") is not None:  # torque ~ density * L^4 at fixed density
            el.set("gear", f"{float(el.get('gear')) * s ** 4:.8g}")
        if el.tag == "joint" and el.get("type") == "slide" and el.get("ref") is not None:
            el.set("ref", f"{float(el.get('ref')) * s:.8g}")
    return ET.tostring(tree.getroot(), encoding="unicode")


if __name__ == "__main__":
    here = __file__.rsplit("/", 1)[0] or "."
    print(scale(f"{here}/walker2d_v5.xml", float(sys.argv[1])))

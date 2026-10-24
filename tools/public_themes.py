"""Display order and themes shared by the homepage and gallery."""

THEMES = (
    "Fibres",
    "Woven composites: plate impact",
    "Woven composites: plain weave",
    "Woven composites: triaxial weave",
    "Woven composites: twill weave",
    "Woven composites: other loading",
    "Metal plasticity and impact",
    "Concrete",
    "Ceramics",
    "Adhesive surface contact",
    "Nonadhesive contact",
    "Fluid dynamics",
    "Fluid-solid interaction",
    "Structural elasticity",
)

HIDDEN_DISPLAY_IDS = frozenset({"contact-models--jkr-footprint", "material-engineering--jc-tension--johnson-cook"})


def theme(value):
    value = value.lower()
    if any(s in value for s in ("crossed-fibre", "single-fibre", "single-fiber", "fibre-pull-off")):
        return THEMES[0]
    if "woven" in value:
        if any(s in value for s in ("plate-impact", "deep-impact", "woven-impact")):
            return THEMES[1]
        if "plain" in value:
            return THEMES[2]
        if "triaxial" in value:
            return THEMES[3]
        if "twill" in value:
            return THEMES[4]
        return THEMES[5]
    if any(s in value for s in (
        "jc-tension", "johnson-cook", "metal-sphere", "metal-impact",
        "metal-linear", "5kms",
    )):
        return THEMES[6]
    if any(s in value for s in ("concrete", "hjc", "rht")):
        return THEMES[7]
    if any(s in value for s in ("ceramic", "jh2")):
        return THEMES[8]
    if any(s in value for s in (
        "adhesion", "adhesive", "pull-off", "sinusoid", "jkr", "dugdale",
        "maugis", "self-consistent", "lateral-release", "three-sphere",
    )):
        return THEMES[9]
    if any(s in value for s in (
        "sphere-contact", "head-on", "oblique", "sliding", "rebound",
        "rubber-ball", "ball-beam", "hertz",
    )):
        return THEMES[10]
    if "unbaffled-tank" in value:
        return THEMES[11]
    if any(s in value for s in (
        "fsi", "elastic-gate", "floating-block", "water-entry", "baffled-tank",
    )):
        return THEMES[12]
    if any(s in value for s in ("tank", "dam-break", "sloshing")):
        return THEMES[11]
    return THEMES[13]


def order(value):
    value = value.lower()
    rank = (
        -1 if "single fibre" in value or "single-fibre" in value or "single fiber" in value
        else 0 if "nonadhesive-pull-off" in value or "nonadhesive pull-off" in value
        else 1 if "adhesive-pull-off" in value or "adhesive pull-off" in value
        else 2 if "sinusoid" in value
        else 3
    )
    return rank, value

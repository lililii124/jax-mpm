"""Shared display themes; does not select or admit public results."""
THEMES=('Metals and high-speed impact','Concrete and ceramics','Woven composites','Adhesion and contact','Fluids and fluid-solid interaction','Structural elasticity')
def theme(value):
 value=value.lower()
 if any(s in value for s in ('jc-tension','johnson-cook','metal-impact','metal-stage','metal-sphere')):return THEMES[0]
 if any(s in value for s in ('concrete','ceramic','hjc','jh2')):return THEMES[1]
 if 'woven' in value:return THEMES[2]
 if any(s in value for s in ('adhesion','adhesive','pull-off','sinusoid','jkr','dugdale','crossed-fibre','sphere-contact','head-on','oblique','sliding','rebound','rubber-ball','ball-beam','lateral-release','three-sphere')):return THEMES[3]
 if any(s in value for s in ('tank','baffle','dam-break','sloshing','fsi','elastic-gate','floating-block','water-entry')):return THEMES[4]
 return THEMES[5]
def order(value):
 value=value.lower()
 rank=0 if 'nonadhesive-pull-off' in value or 'nonadhesive pull-off' in value else 1 if 'adhesive-pull-off' in value or 'adhesive pull-off' in value else 2 if 'sinusoid' in value else 3
 return rank,value

def metal_stage_cell(prefix='docs/'):
 return ('<img src="'+prefix+'images/stages/metal-impact-f16.gif" alt="Metal impact numerical stage preview" width="100%"><br>'
         '<a href="'+prefix+'stages/metal-impact-f16/README.md">Metal impact: f16 numerical stage</a><br>'
         '17 saved states; phase/EOS extrapolation. Not validated 5 km/s physics or a completed protocol.')

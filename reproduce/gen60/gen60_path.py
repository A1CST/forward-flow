"""GEN60 (handoffs/scene_entity_texture/gen60/GEN60_CARD.md): the 60 s shot, one source of truth for the Blender pose/truth render
and the street renderer. World: X along the street (he walks +X), Y up (m), Z across (street centre 0, he walks on the -Z
sidewalk). Per frame: his ground point P, the camera position C and look-at L, and the character VIEW ANGLE a (card convention:
0 profile facing right, 90 back, 180 profile left, 270 front) that the character model must draw.
  python render/gen60_path.py --out <dir>   -> path60.json (+ the Blender batch for the truth / pose render)"""
import argparse, json, math
from pathlib import Path

FPS, SECONDS = 24, 60
WALK_Z = -6.5                     # south sidewalk (road |Z| < 4.5, facades |Z| >= 13.5)
SPEED = 1.57                      # m/s, measured from the stance foot of the tuxedo truth render (6.24 px/frame, 154.5 px head-ankle)
# keyframes: (t_s, azimuth deg of camera around him = atan2(dz, dx) of C - P, distance m, height m, zoom = focal multiplier)
# view angle a = (azimuth - 90) mod 360 (camera at +Z sees +X on its right: he walks screen-right and faces right = 0).
KEYS = [(0, 90, 7.0, 1.4, 1.0), (10, 90, 7.0, 1.4, 1.0),       # side follow from across the street (a = 0, profile facing right)
        (18, 0, 7.5, 1.6, 1.0),                                  # swing to the front: he walks toward the camera (a = 270)
        (21, 0, 7.5, 1.6, 2.6), (25, 0, 7.5, 1.6, 2.6),          # zoom in on the face, hold
        (28, 0, 7.5, 1.6, 1.0),                                  # zoom back out
        (34, 40, 6.0, 2.2, 1.0),                                 # front-quarter, street side, higher (a = 310)
        (42, -45, 4.5, 1.1, 1.0),                                # front-quarter, building side, low (a = 225, interp hold-out band)
        (50, -150, 7.0, 2.6, 1.0),                               # rear-quarter, high (a = 120)
        (60, -270, 7.0, 1.4, 1.0)]                               # through the rear (a = 90) back to the side (a = 0)
# building facades stand at |Z| >= 13.5; main() asserts every camera stays at |Z| < 13


def ease(u): return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, u)))


def at(t):
    for k0, k1 in zip(KEYS, KEYS[1:]):
        if k0[0] <= t <= k1[0]:
            u = ease((t - k0[0]) / max(1e-9, k1[0] - k0[0])); return tuple(x0 + (x1 - x0) * u for x0, x1 in zip(k0[1:], k1[1:]))
    return KEYS[-1][1:]


def frames(speed=SPEED):
    out = []
    for f in range(FPS * SECONDS):
        t = f / FPS; az, dist, hgt, zoom = at(t); P = (speed * t, 0.0, WALK_Z)
        C = (P[0] + dist * math.cos(math.radians(az)), hgt, P[2] + dist * math.sin(math.radians(az)))
        L = (P[0], 1.0 + 0.45 * (zoom - 1.0) / 1.6, P[2]); a = (az - 90.0) % 360.0     # zoom aims at the face
        out.append({"f": f, "t": round(t, 4), "P": [round(v, 4) for v in P], "C": [round(v, 4) for v in C], "L": [round(v, 4) for v in L],
                    "azimuth_deg": round(az, 3), "view_deg": round(a, 3), "zoom": round(zoom, 4)})
    return out


def unwrap(vals):
    o = [vals[0]]
    for v in vals[1:]:
        d = (v - o[-1] + 180) % 360 - 180; o.append(o[-1] + d)
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--speed", type=float, default=SPEED); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); fr = frames(a.speed)
    zmin = min(f["C"][2] for f in fr); zmax = max(f["C"][2] for f in fr); assert -13.0 < zmin and zmax < 13.0, (zmin, zmax)
    (out / "path60.json").write_text(json.dumps({"fps": FPS, "seconds": SECONDS, "speed_mps": a.speed, "walk_z": WALK_Z, "keys": KEYS,
                                                 "convention": "view_deg: 0 profile right, 90 back, 180 profile left, 270 front", "frames": fr}), newline="\n")
    # Blender truth / pose segments: 1 s each, exact linear sweep of the (unwrapped) view angle, one continuous Walk clock
    v = unwrap([x["view_deg"] for x in fr]); segs = []
    for s in range(SECONDS):
        i0, i1 = s * FPS, min(len(v) - 1, (s + 1) * FPS - 1)
        segs.append({"action": "Walk", "frames": i1 - i0 + 1, "facing": {"yaw_from": round(v[i0], 4), "yaw_to": round(v[i1], 4)}})
    (out / "segments60.json").write_text(json.dumps(segs, indent=0), newline="\n")
    print(json.dumps({"frames": len(fr), "view_range": [round(min(v), 1), round(max(v), 1)], "segments": len(segs)}))


if __name__ == "__main__":
    main()

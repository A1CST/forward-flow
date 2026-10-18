# Batch cel-shaded renders, v2: scripted clips with blended actions (Claude 2026-10-17; Payton 1560882287510560859:
# walk, run, squat, jump, "stop + wave to camera").
#   blender -b --factory-startup --python render_batch2.py -- <batch.json>
# clip = {"name", "segments": [{"action": Walk|Run|Idle|Jump|Squat|Wave, "frames": n, "facing": 1|-1|"camera"}], "phase0"}
# Every frame: each asset action is evaluated at its own clock, local bone transforms are captured, and the displayed pose is
# a cross-fade from the previous segment's pose over BLEND frames (no snaps). Squat and Wave are authored on top of Idle
# (actions.py). Facing goes on a parent empty and eases to the target yaw over TURN frames; "camera" = front view.
# Outputs per clip: rgba/<t>.png, labels/<t>.png (flat region pass, gate only), gt.json (13 joints px + world, segment,
# action, yaw). Per character: ref.png, ref_labels.png, ref.json (idle, facing right).
import bpy, json, math, os, sys
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector, Quaternion, Matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import actions as AC
import retarget as RT

B = json.load(open(sys.argv[sys.argv.index("--") + 1])); OUT = os.path.abspath(B["out"]); S = int(B.get("size", 256)); FPS = 24
BLEND, TURN = int(B.get("blend_frames", 8)), int(B.get("turn_frames", 8))
JOINTS = {"head": "Head", "neck": "Neck", "r_shoulder": "RightArm", "r_elbow": "RightForeArm", "r_wrist": "RightHand", "l_shoulder": "LeftArm",
          "l_elbow": "LeftForeArm", "l_wrist": "LeftHand", "pelvis": "Hips", "r_knee": "RightLeg", "r_ankle": "RightFoot", "l_knee": "LeftLeg", "l_ankle": "LeftFoot"}
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.fbx(filepath=B["fbx"])
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE"); mesh = next(o for o in bpy.data.objects if o.type == "MESH")
ACTS = {a.name.split("|")[-1]: a for a in bpy.data.actions}; pb = arm.pose.bones; arm.animation_data_create()
sc = bpy.context.scene; LIGHT_DIR = (0.55, -0.35, 0.76); sc.render.use_persistent_data = True
sc.render.engine = "CYCLES"; sc.cycles.samples = int(B.get("samples", 8)); sc.cycles.use_denoising = False; sc.cycles.max_bounces = 0
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences; prefs.compute_device_type = B.get("device", "CUDA"); prefs.get_devices()
    for dv in prefs.devices: dv.use = True
    sc.cycles.device = "GPU"
except Exception as e:
    print("GPU setup failed, CPU:", e)
sc.render.film_transparent = True; sc.render.resolution_x = sc.render.resolution_y = S; sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"; sc.view_settings.view_transform = "Standard"; sc.render.use_freestyle = False
sc.world = bpy.data.worlds.new("w")
H = max(mesh.dimensions); cam_d = bpy.data.cameras.new("cam"); cam_d.type = "ORTHO"; cam_d.ortho_scale = H * 1.6
cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
cam.location = Vector((H * 4, 0, H * 0.5)); cam.rotation_euler = (math.pi / 2, 0, math.pi / 2)
ROOT = bpy.data.objects.new("facing_root", None); sc.collection.objects.link(ROOT)
for o in (arm, mesh):
    if o.parent is None: o.parent = ROOT
MATS = []
for m in bpy.data.materials:
    if not m.node_tree: continue
    nodes, links = m.node_tree.nodes, m.node_tree.links
    img = next((n for n in nodes if n.type == "TEX_IMAGE"), None) or nodes.new("ShaderNodeTexImage"); img.interpolation = "Closest"
    outn = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    geo = nodes.new("ShaderNodeNewGeometry"); dot = nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"; dot.inputs[1].default_value = LIGHT_DIR
    ramp = nodes.new("ShaderNodeValToRGB"); ramp.color_ramp.interpolation = "CONSTANT"; ramp.color_ramp.elements[0].color = (.62, .62, .68, 1); ramp.color_ramp.elements[1].position = .05
    mul = nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"; mul.inputs["Factor"].default_value = 1.0
    emi = nodes.new("ShaderNodeEmission"); flat = nodes.new("ShaderNodeEmission"); sw = nodes.new("ShaderNodeMixShader"); sw.name = "LABEL_SWITCH"
    links.new(geo.outputs["Normal"], dot.inputs[0]); links.new(dot.outputs["Value"], ramp.inputs["Fac"])
    links.new(img.outputs["Color"], mul.inputs["A"]); links.new(ramp.outputs["Color"], mul.inputs["B"]); links.new(mul.outputs["Result"], emi.inputs["Color"])
    links.new(img.outputs["Color"], flat.inputs["Color"]); links.new(emi.outputs[0], sw.inputs[1]); links.new(flat.outputs[0], sw.inputs[2]); links.new(sw.outputs[0], outn.inputs["Surface"])
    MATS.append((m, img))
PBR = B.get("shading") == "pbr"                                                 # REALISM_CARD R1 (Payton 1561017780990648512)
if PBR:
    for m, img in MATS:
        nodes, links = m.node_tree.nodes, m.node_tree.links; sw = nodes["LABEL_SWITCH"]; img.interpolation = "Linear"
        bsdf = nodes.new("ShaderNodeBsdfPrincipled"); bsdf.inputs["Roughness"].default_value = 0.55
        bsdf.inputs["Specular IOR Level"].default_value = 0.3; links.new(img.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(bsdf.outputs[0], sw.inputs[1])
    for poly in mesh.data.polygons: poly.use_smooth = True
    sub = mesh.modifiers.new("subd", "SUBSURF"); sub.levels = sub.render_levels = 1
    for nm, loc, en, sz in (("key", (H * 2.5, -H * 2.0, H * 2.6), 900.0, H * 1.2), ("fill", (H * 2.5, H * 2.2, H * 0.9), 260.0, H * 1.6),
                            ("rim", (-H * 2.5, -H * 0.8, H * 1.8), 520.0, H * 0.8)):
        ld = bpy.data.lights.new(nm, "AREA"); ld.energy = en * H * H * float(B.get("light_scale", 1.0)); ld.size = sz; lo = bpy.data.objects.new(nm, ld)
        sc.collection.objects.link(lo); lo.location = Vector(loc); lo.rotation_euler = (Vector((0, 0, H * .5)) - lo.location).to_track_quat("-Z", "Y").to_euler()
    sc.world.use_nodes = True; sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.25
    sc.cycles.max_bounces = 4; sc.cycles.diffuse_bounces = 3; sc.cycles.glossy_bounces = 2; sc.cycles.transparent_max_bounces = 8


def set_look(ch, label=False):
    for m, img in MATS:
        m.node_tree.nodes["LABEL_SWITCH"].inputs[0].default_value = 1.0 if label else 0.0
        img.image = bpy.data.images.load(ch["label_texture" if label else "texture"], check_existing=True)
        if PBR: img.interpolation = "Closest" if label else "Linear"
    sc.render.filter_size = 0.0 if label else 1.5; sc.cycles.samples = 1 if label else int(B.get("samples", 8))
    if PBR:                                                                     # label pass stays flat emission in Standard view, no denoise
        sc.cycles.use_denoising = not label; sc.view_settings.view_transform = "Standard" if label else B.get("view", "AgX")


MOCAP = {}                                                                     # Payton 1560886201718931488: the wave is CMU 141_16
PLANT = {n for n in pb.keys() if n == "Hips" or any(k in n for k in ("UpLeg", "Leg", "Foot", "Toe"))}


def capture(action, frame):
    if action in MOCAP:                                                         # retargeted capture, posed in the root's rest yaw
        R, fr, yaw0 = MOCAP[action]; keep = ROOT.rotation_euler.z; ROOT.rotation_euler.z = yaw0; bpy.context.view_layer.update()
        arm.animation_data.action = None; R.apply(fr[int(frame) % len(fr)])
        out = (arm.matrix_basis.copy(), {b.name: (b.location.copy(), b.rotation_quaternion.copy(), b.scale.copy()) for b in pb})
        if B.get("wave_plant"):                                                 # Payton 1561216783745032254: legs float during the wave.
            ROOT.rotation_euler.z = keep; bpy.context.view_layer.update()       # Hips + legs + object transform held at Idle frame 0:
            im, ib = capture("Idle", 0); ROOT.rotation_euler.z = yaw0           # feet planted, only the upper body plays the capture.
            out = (im, {k: (ib[k] if k in PLANT else v) for k, v in out[1].items()})
        ROOT.rotation_euler.z = keep; bpy.context.view_layer.update(); return out
    a = ACTS[action]; arm.animation_data.action = a
    if hasattr(arm.animation_data, "action_slot") and a.slots: arm.animation_data.action_slot = a.slots[0]
    f0, f1 = a.frame_range; L = max(1.0, f1 - f0); sc.frame_set(int(f0 + frame % L))
    return arm.matrix_basis.copy(), {b.name: (b.location.copy(), b.rotation_quaternion.copy(), b.scale.copy()) for b in pb}


def apply(objm, pose):
    arm.animation_data.action = None; arm.matrix_basis = objm
    for b in pb:
        l, q, s = pose[b.name]; b.rotation_mode = "QUATERNION"; b.location, b.rotation_quaternion, b.scale = l, q, s
    bpy.context.view_layer.update()


def mix(pa, pb_, w):
    (ma, a), (mb, b) = pa, pb_
    out = {k: (a[k][0].lerp(b[k][0], w), a[k][1].slerp(b[k][1], w), a[k][2].lerp(b[k][2], w)) for k in a}
    la, ra, sa = ma.decompose(); lb, rb, sb = mb.decompose()
    return Matrix.LocRotScale(la.lerp(lb, w), ra.slerp(rb, w), sa.lerp(sb, w)), out


BASE_OF = {"Walk": "Walk", "Run": "Run", "Idle": "Idle", "Jump": "Jump", "Squat": "Idle", "Wave": "WaveMocap" if B.get("wave_bvh") else "Idle"}


def seg_pose(seg, u, clock):
    """Pose (objm, bones) of one segment at local frame u, plus a flag for the authored extras."""
    return capture(BASE_OF[seg["action"]], clock)


def joints():
    jp, jw = [], []
    for k, bn in JOINTS.items():
        b = pb[bn]; p = arm.matrix_world @ (b.tail if k == "head" else b.head)
        v = world_to_camera_view(sc, cam, p); jp.append([round(v.x * S, 2), round((1 - v.y) * S, 2)]); jw.append([round(p.y, 4), round(p.z, 4)])
    return jp, jw


def yaw_for(f):
    return B.get("right_yaw", math.pi) + (-math.pi / 2 if f == "camera" else (0.0 if f > 0 else math.pi))


if B.get("wave_bvh"):
    ROOT.rotation_euler.z = yaw_for("camera") + float(os.environ.get("MOCAP_YAW_OFF", "0")); bpy.context.view_layer.update()
    _R = RT.Retarget(arm, B["wave_bvh"], bpy.context, extra_yaw=float(os.environ.get("MOCAP_EXTRA_YAW", B.get("wave_extra_yaw", 0.0)))); MOCAP["WaveMocap"] = (_R, _R.frames(FPS)[2:], ROOT.rotation_euler.z)


def render(path):
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)


for ch in B["characters"]:
    cdir = os.path.join(OUT, ch["name"]); os.makedirs(cdir, exist_ok=True)
    pb["Head"].scale = Vector([ch.get("head_scale", 1.0)] * 3)
    ROOT.rotation_euler.z = yaw_for(1); apply(*capture("Idle", 0)); pb["Head"].scale = Vector([ch.get("head_scale", 1.0)] * 3); bpy.context.view_layer.update()
    up = (pb["Head"].head - pb["Hips"].head).normalized(); arm["idle_foot_low"] = min(pb["RightFoot"].head.dot(up), pb["LeftFoot"].head.dot(up))
    set_look(ch); render(os.path.join(cdir, "ref.png")); set_look(ch, True); render(os.path.join(cdir, "ref_labels.png"))
    rp, rw = joints(); json.dump({"pose": "idle frame 0, facing right", "joints_px": rp, "joints_world": rw}, open(os.path.join(cdir, "ref.json"), "w"))
    for cl in ch["clips"]:
        d = os.path.join(cdir, cl["name"]); os.makedirs(os.path.join(d, "rgba"), exist_ok=True); os.makedirs(os.path.join(d, "labels"), exist_ok=True)
        recs = []; t = 0; prev = None; yaw = yaw_for(cl["segments"][0]["facing"]); clocks = {}
        for si, seg in enumerate(cl["segments"]):
            for u in range(seg["frames"]):
                base = BASE_OF[seg["action"]]; clocks[base] = (clocks.get(base, cl.get("phase0", 0.0) * 25) + 1) if base not in MOCAP else u
                cur = capture(base, clocks[base])
                w = min(1.0, (u + 1) / BLEND) if (prev is not None and u < BLEND) else 1.0
                pose = mix(prev, cur, w) if w < 1.0 else cur
                apply(*pose); pb["Head"].scale = Vector([ch.get("head_scale", 1.0)] * 3); bpy.context.view_layer.update()
                if seg["action"] == "Squat":
                    AC.squat(arm, bpy.context, math.sin(math.pi * u / max(1, seg["frames"] - 1)) ** 2)
                elif seg["action"] == "Wave" and not B.get("wave_bvh"):
                    (AC.wave2 if B.get("wave", "wave2") == "wave2" else AC.wave)(arm, bpy.context, min(1.0, u / 8.0), 2 * math.pi * 2.0 * u / FPS)
                target = yaw_for(seg["facing"]); yaw += max(-math.pi / TURN, min(math.pi / TURN, target - yaw)); ROOT.rotation_euler.z = yaw
                bpy.context.view_layer.update()
                set_look(ch); render(os.path.join(d, "rgba", f"{t:05d}.png")); set_look(ch, True); render(os.path.join(d, "labels", f"{t:05d}.png"))
                jp, jw = joints(); recs.append({"t": t, "joints_px": jp, "joints_world": jw, "segment": si, "action": seg["action"],
                                                "facing": seg["facing"], "yaw": round(yaw, 4), "blend_w": round(w, 3)})
                t += 1
            prev = (arm.matrix_basis.copy(), {b.name: (b.location.copy(), b.rotation_quaternion.copy(), b.scale.copy()) for b in pb})   # last shown pose (incl. extras)
        json.dump({"schema": "ff_cartoon_blender_v2", "fps": FPS, "character": ch, "clip": cl, "frames": recs}, open(os.path.join(d, "gt.json"), "w"))
    print("CHAR_OK", ch["name"], flush=True)
print("BATCH_OK", len(B["characters"]))

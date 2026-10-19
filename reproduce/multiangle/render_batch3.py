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
JOINTS = B.get("joints_map", JOINTS)                                           # EVAL_T1_T3_CARD T1c: another rig (Knight) maps its own bones
ACT_ALIAS = B.get("action_map", {})                                            # e.g. {"Walk": "Walking"} for the Knight
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
if B.get("cpu_render"):                                                         # SPEED rev 6 (Payton 1561541734524125195): CPU-only render pods, output passed to a GPU pod
    sc.cycles.device = "CPU"
print("RENDER_DEVICE", sc.cycles.device)
if B.get("threads"):                                                            # SPEED rev 5: N parallel workers on a cgroup quota; Cycles' default = one thread per HOST cpu (64) thrashed 8 workers
    sc.render.threads_mode = "FIXED"; sc.render.threads = int(B["threads"])
sc.render.film_transparent = True; sc.render.resolution_x = sc.render.resolution_y = S; sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"; sc.view_settings.view_transform = "Standard"; sc.render.use_freestyle = False
sc.world = bpy.data.worlds.new("w")
H = max(mesh.dimensions); cam_d = bpy.data.cameras.new("cam"); cam_d.type = "ORTHO"; cam_d.ortho_scale = H * 1.6
cam = bpy.data.objects.new("cam", cam_d); sc.collection.objects.link(cam); sc.camera = cam
cam.location = Vector((H * 4, 0, H * 0.5)); cam.rotation_euler = (math.pi / 2, 0, math.pi / 2)
ROOT = bpy.data.objects.new("facing_root", None); sc.collection.objects.link(ROOT)
for o in (arm, mesh):
    if o.parent is None: o.parent = ROOT
STRIPE = None                                                                   # T3 (opt-in): shirt faces in a height band get their own material
if any(c.get("stripe") for c in B["characters"]):
    base_m = mesh.data.materials[0]; smat = base_m.copy(); smat.name = "STRIPE_MAT"; mesh.data.materials.append(smat); s_idx = len(mesh.data.materials) - 1
    lay = bpy.data.images.load(B["layout_texture"]); lw, lh = lay.size; px = list(lay.pixels); uvl = mesh.data.uv_layers.active.data
    def _col(u, v):
        x = min(lw - 1, max(0, int(u * lw))); y = min(lh - 1, max(0, int(v * lh))); i = 4 * (y * lw + x); return [round(px[i + k] * 255) for k in range(3)]
    shirt = [p for p in mesh.data.polygons if _col(sum(uvl[l].uv[0] for l in p.loop_indices) / p.loop_total, sum(uvl[l].uv[1] for l in p.loop_indices) / p.loop_total) == [115, 167, 196]]
    zs = [p.center.z for p in shirt]; z0, z1 = min(zs), max(zs)
    STRIPE = {"idx": s_idx, "orig": {p.index: p.material_index for p in shirt}, "shirt": shirt, "z0": z0, "z1": z1}
FAMS = ["hbands", "vbands", "diagonal", "checks", "plaid", "dots", "logo", "blocks", "noise"]   # 'solid' = all weights 0
PAT_LABNODES = []
PAT_AXES = {}
LABMASK = {}
IMG_SRGB = {}


def pattern_nodes(m, img):
    """Per-pixel procedural shirt in rest-pose (Generated) coordinates. Shirt pixels = label texture grey 120.
    Binary families give an ink mask in {0,1}: colour = mix(c0, c1, ink); label pass = 120 + 5*ink (125 = ink).
    blocks / noise are multi-colour (label 120 only). Parameters live in named Value/RGB nodes, set per character."""
    N, L = m.node_tree.nodes, m.node_tree.links

    def node(t, **kw):
        n = N.new(t)
        for k, v in kw.items(): setattr(n, k, v)
        return n

    def val(name, v=0.0):
        n = node("ShaderNodeValue", name=name); n.outputs[0].default_value = v; return n.outputs[0]

    def rgb(name):
        return node("ShaderNodeRGB", name=name).outputs[0]

    def math(op, a, b=None):
        n = node("ShaderNodeMath", operation=op)
        if isinstance(a, float): n.inputs[0].default_value = a
        else: L.new(a, n.inputs[0])
        if b is not None:
            if isinstance(b, float): n.inputs[1].default_value = b
            else: L.new(b, n.inputs[1])
        return n.outputs[0]

    def mixc(a, b, f):
        n = node("ShaderNodeMix", data_type="RGBA"); L.new(f, n.inputs["Factor"]); L.new(a, n.inputs[6]); L.new(b, n.inputs[7]); return n.outputs[2]

    lab = node("ShaderNodeTexImage", name="PAT_LABTEX", interpolation="Closest"); L.new(N.new("ShaderNodeUVMap").outputs[0], lab.inputs[0])
    PAT_LABNODES.append(lab)
    lsep = node("ShaderNodeSeparateColor"); L.new(lab.outputs["Color"], lsep.inputs[0])
    shirt = math("LESS_THAN", math("ABSOLUTE", math("SUBTRACT", lsep.outputs[0], 120 / 255)), 0.03)
    tc = node("ShaderNodeTexCoord"); sep = node("ShaderNodeSeparateXYZ"); L.new(tc.outputs["Generated"], sep.inputs[0])
    Mw = mesh.matrix_world.to_3x3(); up = max(range(3), key=lambda i: abs(Mw[2][i])); hz = [i for i in range(3) if i != up]
    z = sep.outputs[up]                                                            # body-vertical axis (world up)
    x = math("ADD", sep.outputs[hz[0]], sep.outputs[hz[1]])                        # both horizontal body axes: varies in profile AND front view
    PAT_AXES.update({"up": up, "horizontal": hz})
    sc, ph = val("PAT_SCALE", 8.0), val("PAT_PHASE", 0.0)
    band = lambda u: math("LESS_THAN", math("FRACT", math("ADD", math("MULTIPLY", u, sc), ph)), 0.5)
    mh, mv, md = band(z), band(x), band(math("ADD", x, z))
    mc = math("MODULO", math("ADD", math("FLOOR", math("MULTIPLY", x, sc)), math("FLOOR", math("MULTIPLY", z, sc))), 2.0)
    mp = math("MAXIMUM", band(x), band(z))                                         # plaid = union of two band sets (binary)
    dx = math("SUBTRACT", math("FRACT", math("MULTIPLY", x, sc)), 0.5); dz = math("SUBTRACT", math("FRACT", math("MULTIPLY", z, sc)), 0.5)
    mdot = math("LESS_THAN", math("SQRT", math("ADD", math("MULTIPLY", dx, dx), math("MULTIPLY", dz, dz))), 0.3)
    inr = lambda u, lo, hi: math("MULTIPLY", math("GREATER_THAN", u, val(lo, 0.3)), math("LESS_THAN", u, val(hi, 0.7)))
    mlogo = math("MULTIPLY", inr(x, "PAT_LX0", "PAT_LX1"), inr(z, "PAT_LZ0", "PAT_LZ1"))
    w = {f: val("PAT_W_" + f) for f in FAMS}
    ink = val("PAT_ZERO", 0.0)
    for f, mm in (("hbands", mh), ("vbands", mv), ("diagonal", md), ("checks", mc), ("plaid", mp), ("dots", mdot), ("logo", mlogo)):
        ink = math("ADD", ink, math("MULTIPLY", w[f], mm))
    c0, c1, c2, c3 = rgb("PAT_C0"), rgb("PAT_C1"), rgb("PAT_C2"), rgb("PAT_C3")
    binary = mixc(c0, c1, ink)
    vco = node("ShaderNodeVectorMath", operation="SCALE"); L.new(tc.outputs["Generated"], vco.inputs[0]); L.new(sc, vco.inputs["Scale"])
    vor = node("ShaderNodeTexVoronoi", distance="CHEBYCHEV"); L.new(vco.outputs[0], vor.inputs["Vector"])
    vsep = node("ShaderNodeSeparateColor"); L.new(vor.outputs["Color"], vsep.inputs[0]); rr = vsep.outputs[0]
    blk = mixc(mixc(c0, c1, math("GREATER_THAN", rr, 0.25)), mixc(c2, c3, math("GREATER_THAN", rr, 0.75)), math("GREATER_THAN", rr, 0.5))

    def snapnoise(s, seedname):
        v = node("ShaderNodeVectorMath", operation="SCALE"); L.new(tc.outputs["Generated"], v.inputs[0]); L.new(s, v.inputs["Scale"])
        fl = node("ShaderNodeVectorMath", operation="FLOOR"); L.new(v.outputs[0], fl.inputs[0])
        sd = val(seedname, 0.0); cv = node("ShaderNodeCombineXYZ")
        for i in range(3): L.new(sd, cv.inputs[i])
        off = node("ShaderNodeVectorMath", operation="ADD"); L.new(fl.outputs[0], off.inputs[0]); L.new(cv.outputs[0], off.inputs[1])
        wn = node("ShaderNodeTexWhiteNoise", noise_dimensions="3D"); L.new(off.outputs[0], wn.inputs["Vector"]); return wn.outputs["Color"]

    noi = mixc(snapnoise(sc, "PAT_SEED_A"), snapnoise(math("MULTIPLY", sc, 3.0), "PAT_SEED_B"), val("PAT_HALF", 0.35))
    col = mixc(mixc(binary, blk, w["blocks"]), noi, w["noise"])
    g = math("DIVIDE", math("ADD", 120.0, math("MULTIPLY", ink, 5.0)), 255.0)        # ink label 125: same post_outline band as shirt 120 (no stripe outlines)
    g = math("POWER", math("DIVIDE", math("ADD", g, 0.055), 1.055), 2.4)          # sRGB -> linear, so the Standard view writes exactly 120 / 125
    labgrey = node("ShaderNodeCombineColor")
    for i in range(3): L.new(g, labgrey.inputs[i])
    shirt_col = mixc(col, labgrey.outputs[0], val("PAT_LABMODE", 0.0))
    return mixc(img.outputs["Color"], shirt_col, shirt)


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
    col_out = pattern_nodes(m, img) if B.get("patterns") else img.outputs["Color"]      # PATTERN_SHIRT_CARD (opt-in)
    links.new(col_out, mul.inputs["A"]); links.new(ramp.outputs["Color"], mul.inputs["B"]); links.new(mul.outputs["Result"], emi.inputs["Color"])
    links.new(col_out, flat.inputs["Color"]); links.new(emi.outputs[0], sw.inputs[1]); links.new(flat.outputs[0], sw.inputs[2]); links.new(sw.outputs[0], outn.inputs["Surface"])
    MATS.append((m, img))
TRACK = []; TRACK_FACES = []                                                   # PATTERN_SHIRT_CARD texture-lock check: fixed shirt vertices (or face centroids)
if B.get("patterns"):
    _lay = bpy.data.images.load(B["layout_texture"]); _lw, _lh = _lay.size; _px = list(_lay.pixels); _uvl = mesh.data.uv_layers.active.data
    def _c(u, v):
        i = 4 * (min(_lh - 1, max(0, int(v * _lh))) * _lw + min(_lw - 1, max(0, int(u * _lw)))); return [round(_px[i + k] * 255) for k in range(3)]
    _sv = sorted({vi for p_ in mesh.data.polygons if _c(sum(_uvl[l].uv[0] for l in p_.loop_indices) / p_.loop_total, sum(_uvl[l].uv[1] for l in p_.loop_indices) / p_.loop_total) == [115, 167, 196] for vi in p_.vertices})
    TRACK = _sv[:: max(1, len(_sv) // 24)][:24]
    TRACK_FACES = []
    if B.get("track_mode") == "face":                                           # texture-lock repair: interior points (face centroids), not vertices
        _sf = [p_.index for p_ in mesh.data.polygons if _c(sum(_uvl[l].uv[0] for l in p_.loop_indices) / p_.loop_total, sum(_uvl[l].uv[1] for l in p_.loop_indices) / p_.loop_total) == [115, 167, 196]]
        TRACK_FACES = _sf[:: max(1, len(_sf) // 48)][:48]
    _bb = [mesh.data.vertices[0].co.copy(), mesh.data.vertices[0].co.copy()]
    for _v in mesh.data.vertices:
        for _i in range(3): _bb[0][_i] = min(_bb[0][_i], _v.co[_i]); _bb[1][_i] = max(_bb[1][_i], _v.co[_i])
    _g = lambda co, i: (co[i] - _bb[0][i]) / max(1e-9, _bb[1][i] - _bb[0][i])
    print("PAT_SHIRT_RANGE", PAT_AXES, [[round(min(_g(mesh.data.vertices[v].co, i) for v in _sv), 3), round(max(_g(mesh.data.vertices[v].co, i) for v in _sv), 3)] for i in range(3)], flush=True)


def track_px():
    if not TRACK: return None
    dg = bpy.context.evaluated_depsgraph_get(); ev = mesh.evaluated_get(dg); me = ev.to_mesh(); out = []
    if TRACK_FACES:
        for fi in TRACK_FACES:
            co = sum((me.vertices[vi].co for vi in me.polygons[fi].vertices), Vector()) / len(me.polygons[fi].vertices)
            v = world_to_camera_view(sc, cam, ev.matrix_world @ co); out.append([round(v.x * S, 2), round((1 - v.y) * S, 2), round(v.z, 4)])
    else:
        for vi in TRACK:
            v = world_to_camera_view(sc, cam, ev.matrix_world @ me.vertices[vi].co); out.append([round(v.x * S, 2), round((1 - v.y) * S, 2), round(v.z, 4)])
    ev.to_mesh_clear(); return out


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


def flat_image(name, rgb):
    im = bpy.data.images.get(name) or bpy.data.images.new(name, 1, 1); im.pixels = [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0]; return im


def set_look(ch, label=False):
    for m, img in MATS:
        m.node_tree.nodes["LABEL_SWITCH"].inputs[0].default_value = 1.0 if label else 0.0
        if m.name == "STRIPE_MAT":
            st = ch.get("stripe") or {"rgb": [255, 0, 255], "label": 230}
            img.image = flat_image(f"{ch['name']}_stripe_{int(label)}", [st["label"]] * 3 if label else st["rgb"]); continue
        if B.get("patterns"):                                                  # PATTERN_SHIRT_CARD: per-character shirt parameters
            nd = m.node_tree.nodes; sp = ch.get("shirt", {"family": "solid", "colors": [[128, 128, 128]]})
            li = LABMASK.get(ch["label_texture"])                                   # separate datablock: raw greys for the shirt mask (Non-Color);
            if li is None:                                                         # the label-pass image stays sRGB so label greys are unchanged
                li = bpy.data.images.load(ch["label_texture"], check_existing=False); li.colorspace_settings.name = "Non-Color"; LABMASK[ch["label_texture"]] = li
            nd["PAT_LABTEX"].image = li
            nd["PAT_LABMODE"].outputs[0].default_value = 1.0 if label else 0.0
            for f in FAMS: nd["PAT_W_" + f].outputs[0].default_value = 1.0 if sp["family"] == f else 0.0
            cs = sp["colors"] + [sp["colors"][-1]] * (4 - len(sp["colors"]))
            for i in range(4): nd[f"PAT_C{i}"].outputs[0].default_value = (cs[i][0] / 255, cs[i][1] / 255, cs[i][2] / 255, 1.0)
            nd["PAT_SCALE"].outputs[0].default_value = float(sp.get("scale", 8.0)); nd["PAT_PHASE"].outputs[0].default_value = float(sp.get("phase", 0.0))
            nd["PAT_SEED_A"].outputs[0].default_value = float(sp.get("seed", 0)); nd["PAT_SEED_B"].outputs[0].default_value = float(sp.get("seed", 0)) + 0.5
            for k, v in zip(("PAT_LX0", "PAT_LX1", "PAT_LZ0", "PAT_LZ1"), sp.get("logo", [0.35, 0.65, 0.45, 0.7])): nd[k].outputs[0].default_value = float(v)
        if ch.get("material_colors"):                                          # T1c: flat per-material colours (rig without one texture)
            key = m.name.split(".")[0]; rgb = (ch["material_labels"] if label else ch["material_colors"]).get(key, [128, 128, 128])
            if label: rgb = [rgb, rgb, rgb] if not isinstance(rgb, list) else rgb
            img.image = flat_image(f"{ch['name']}_{key}_{int(label)}", rgb); continue
        path = ch["label_texture" if label else "texture"]
        if B.get("patterns"):                                                  # own cache so the Non-Color mask copy can never be picked up here
            if path not in IMG_SRGB: IMG_SRGB[path] = bpy.data.images.load(path, check_existing=False)
            img.image = IMG_SRGB[path]
        else:
            img.image = bpy.data.images.load(path, check_existing=True)
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
    a = ACTS[ACT_ALIAS.get(action, action)]; arm.animation_data.action = a
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
    if isinstance(f, dict):                                                     # MULTIANGLE card: {"yaw_deg": a} fixed, or {"yaw_from": a0, "yaw_to": a1} sweep
        return B.get("right_yaw", math.pi) + math.radians(float(f.get("yaw_deg", f.get("yaw_from", 0.0))))   # a = 0 profile facing right, 270 facing camera
    return B.get("right_yaw", math.pi) + (-math.pi / 2 if f == "camera" else (0.0 if f > 0 else math.pi))


if B.get("wave_bvh"):
    ROOT.rotation_euler.z = yaw_for("camera") + float(os.environ.get("MOCAP_YAW_OFF", "0")); bpy.context.view_layer.update()
    _R = RT.Retarget(arm, B["wave_bvh"], bpy.context, extra_yaw=float(os.environ.get("MOCAP_EXTRA_YAW", B.get("wave_extra_yaw", 0.0)))); MOCAP["WaveMocap"] = (_R, _R.frames(FPS)[2:], ROOT.rotation_euler.z)


LEG_BONES = B.get("leg_bones", ["LeftUpLeg", "RightUpLeg"])


def legs(ch):
    """T1b leg-long: scale the thigh along its length (the shin inherits it); lift the root so the feet stay on the ground."""
    s = float(ch.get("leg_scale", 1.0))
    if s == 1.0: return
    for n in LEG_BONES:
        if n in pb: pb[n].scale = Vector((1.0, s, 1.0))
    lb = arm.data.bones[LEG_BONES[0]]; L = lb.length + (lb.children[0].length if lb.children else 0.0)
    ROOT.location.z = (s - 1.0) * L * arm.matrix_world.to_scale()[2]


def proportions(ch):
    """T1b audit (Astra 1561235583240773696): the exact transforms in force, logged for the ref and every target frame."""
    if "body_scale" not in ch and "leg_scale" not in ch: return {}
    return {"proportions": {"root_scale": [round(v, 4) for v in ROOT.scale], "root_z": round(ROOT.location.z, 5), "leg_scale": float(ch.get("leg_scale", 1.0)),
                            "leg_bone_scale": {n: [round(v, 4) for v in pb[n].scale] for n in LEG_BONES if n in pb}, "head_scale": [round(v, 4) for v in pb["Head"].scale]}}


FAST = bool(B.get("fast"))                                                       # OPTION-1 card: opt-in speed path (default = v2 behaviour exactly)
_ENG = {}


def _eevee_id():
    if "id" not in _ENG:
        for e in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try: sc.render.engine = e; _ENG["id"] = e; break
            except Exception: pass
        sc.render.engine = "CYCLES"
    return _ENG.get("id")


def render(path, data_pass=False):
    """data_pass + FAST: the flat 1-sample label / surface / part images are drawn by EEVEE (GPU rasterizer); rgba stays Cycles."""
    sc.render.filepath = path
    if B.get("fast_eevee") and data_pass and _eevee_id():                       # measured 2x SLOWER locally (engine switch re-sync); off by default
        sc.render.engine = _eevee_id(); ee = sc.eevee
        for k, v in (("taa_render_samples", 1), ("use_gtao", False), ("use_bloom", False), ("use_ssr", False), ("use_shadows", False)):
            if hasattr(ee, k): setattr(ee, k, v)
        bpy.ops.render.render(write_still=True); sc.render.engine = "CYCLES"
    else:
        bpy.ops.render.render(write_still=True)


SURFMAT = None                                                                  # UV_TEXTURE_CARD stage 1 (opt-in "surface_pass"): renderer-side
TIMES = {"rgba": [], "labels": [], "surf": [], "part": []}                      # body-surface coordinate pass (supervision / diagnostics only)
PARTMAT = None
PART_NAMES = ["none", "head", "torso", "l_upper_arm", "l_forearm_hand", "r_upper_arm", "r_forearm_hand", "l_thigh", "l_shin_foot", "r_thigh", "r_shin_foot"]


def part_of_bone(n):
    """Body-part ID from a vertex-group (bone) name of the Quaternius rig; 0 = unknown."""
    side = "l_" if n.startswith("Left") else ("r_" if n.startswith("Right") else "")
    core = n[4:] if side == "l_" else (n[5:] if side == "r_" else n)
    if not side: return 1 if core.startswith(("Neck", "Head")) else (2 if core.startswith(("Hips", "Spine")) else 0)
    if core.startswith(("Shoulder", "Arm")): return PART_NAMES.index(side + "upper_arm")
    if core.startswith(("ForeArm", "Hand")): return PART_NAMES.index(side + "forearm_hand")
    if core.startswith("UpLeg"): return PART_NAMES.index(side + "thigh")
    if core.startswith(("Leg", "Foot", "Toe")): return PART_NAMES.index(side + "shin_foot")
    return 0


FACE_PART = []


def build_parts():
    """Per-face part ID = part of the vertex group with the largest summed weight over the face's vertices; stored as a
    FACE-domain colour attribute "part_id" (R = id / 255, G = face index / 65535), so the pass is flat inside every face
    (no interpolation) and every pixel also names its face (v3: per-pixel projected face area via face_area_px)."""
    global FACE_PART
    me = mesh.data; gname = {g.index: g.name for g in mesh.vertex_groups}; FACE_PART = []
    for p in me.polygons:
        acc = {}
        for vi in p.vertices:
            for g in me.vertices[vi].groups:
                k = part_of_bone(gname.get(g.group, "")); acc[k] = acc.get(k, 0.0) + g.weight
        acc.pop(0, None); FACE_PART.append(max(acc, key=acc.get) if acc else 0)
    at = me.attributes.get("part_id") or me.attributes.new("part_id", "FLOAT_COLOR", "FACE")
    for i, k in enumerate(FACE_PART): at.data[i].color = (k / 255.0, i / 65535.0, 0.0, 1.0)


def part_render(path):
    """Fourth render: emission = face attribute part_id, same Raw / 16-bit / 1-sample / ~0-filter settings as the surface
    pass (restored afterwards)."""
    global PARTMAT
    if PARTMAT is None:
        PARTMAT = bpy.data.materials.new("PART_PASS"); PARTMAT.use_nodes = True; nt = PARTMAT.node_tree; nt.nodes.clear()
        at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_type = "GEOMETRY"; at.attribute_name = "part_id"
        em = nt.nodes.new("ShaderNodeEmission"); out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(at.outputs["Color"], em.inputs["Color"]); em.inputs["Strength"].default_value = 1.0; nt.links.new(em.outputs[0], out.inputs["Surface"])
    pass_render(PARTMAT, path)


def pass_render(mat, path):
    vs, im = sc.view_settings, sc.render.image_settings; vl = bpy.context.view_layer
    keep = (vs.view_transform, vs.look, vs.exposure, vs.gamma, sc.render.filter_size, sc.cycles.samples, sc.cycles.use_denoising, im.color_depth, vl.material_override)
    vl.material_override = mat; vs.view_transform = "Raw"; vs.look = "None"; vs.exposure = 0.0; vs.gamma = 1.0
    sc.render.filter_size = 0.0; sc.cycles.samples = 1; sc.cycles.use_denoising = False; im.color_depth = "16"
    render(path, data_pass=True)
    vs.view_transform, vs.look, vs.exposure, vs.gamma, sc.render.filter_size, sc.cycles.samples, sc.cycles.use_denoising, im.color_depth, vl.material_override = keep


def surf_render(path):
    """Third render per frame with a view-layer material override: emission = the Generated coordinate (rest-pose object
    space normalised by the mesh texture space, in [0, 1]^3). It is computed from the UNDEFORMED mesh, so a surface point
    keeps its value under armature deformation, head/leg bone scaling and root scale. Front and back of the body differ
    in the horizontal depth axis, so they never share a value. Raw view, 16-bit PNG, 1 sample, ~0 pixel filter (each
    pixel = one surface point). Every setting it touches is restored afterwards, so the rgba / label passes are unchanged."""
    global SURFMAT
    if SURFMAT is None:
        SURFMAT = bpy.data.materials.new("SURF_PASS"); SURFMAT.use_nodes = True; nt = SURFMAT.node_tree; nt.nodes.clear()
        tc = nt.nodes.new("ShaderNodeTexCoord"); em = nt.nodes.new("ShaderNodeEmission"); out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tc.outputs["Generated"], em.inputs["Color"]); em.inputs["Strength"].default_value = 1.0; nt.links.new(em.outputs[0], out.inputs["Surface"])
    pass_render(SURFMAT, path)


def timed(kind, fn, *a):
    import time
    t0 = time.time(); fn(*a); TIMES[kind].append(time.time() - t0)


def gen_of(co):
    """Generated coordinate of a REST-pose mesh-space point: (co - (texspace_location - texspace_size)) / (2 * texspace_size)."""
    me = mesh.data; L, Sz = me.texspace_location, me.texspace_size
    return [(co[i] - (L[i] - Sz[i])) / (2 * Sz[i]) for i in range(3)]


def surf_points(path, n=400000):
    """Off-surface check data: n area-weighted random points on the REST mesh surface, in Generated space (.npy)."""
    import numpy as np
    me = mesh.data; me.calc_loop_triangles(); T = np.array([[gen_of(me.vertices[v].co) for v in t.vertices] for t in me.loop_triangles])
    ar = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1); idx = np.random.default_rng(0).choice(len(T), n, p=ar / ar.sum())
    r1, r2 = np.random.default_rng(1).random((2, n)); s1 = np.sqrt(r1); w0, w1, w2 = 1 - s1, s1 * (1 - r2), s1 * r2
    P = w0[:, None] * T[idx, 0] + w1[:, None] * T[idx, 1] + w2[:, None] * T[idx, 2]; np.save(path, P.astype(np.float32))
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); N = N / np.linalg.norm(N, axis=1, keepdims=True).clip(1e-12)          # rest normal (Generated axes keep sign)
    np.save(path.replace(".npy", "_normals.npy"), N[idx].astype(np.float32))
    np.save(path.replace(".npy", "_part.npy"), np.array([FACE_PART[t.polygon_index] for t in me.loop_triangles], np.uint8)[idx])
    if B.get("patterns"):                                                       # shirt flag: same layout-colour rule as the texture-lock faces
        uvl = me.uv_layers.active.data; sh = np.array([_c(sum(uvl[l].uv[0] for l in t.loops) / 3, sum(uvl[l].uv[1] for l in t.loops) / 3) == [115, 167, 196] for t in me.loop_triangles])
        np.save(path.replace(".npy", "_shirt.npy"), sh[idx])


def surf_samples():
    """Numeric reconstruction data: fixed faces (every k-th polygon) that face the camera in this frame -> render-time
    pixel (x, y) of the deformed face centroid, the face's REST-pose Generated centroid, face id, part id, and a
    visibility flag (a camera ray to the centroid hits this same face first = not occluded)."""
    dg = bpy.context.evaluated_depsgraph_get(); ev = mesh.evaluated_get(dg); me = ev.to_mesh(); out = []
    cam_dir = (cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized(); mw = ev.matrix_world; nmat = mw.to_3x3().inverted().transposed()
    for fi in SURF_FACES:
        p = me.polygons[fi]
        if (nmat @ p.normal).normalized().dot(cam_dir) > -0.3: continue           # back-facing / grazing: skipped
        co = sum((me.vertices[v].co for v in p.vertices), Vector()) / len(p.vertices); v = world_to_camera_view(sc, cam, mw @ co)
        rp = mesh.data.polygons[fi]; rc = sum((mesh.data.vertices[v_].co for v_ in rp.vertices), Vector()) / len(rp.vertices)
        wp = mw @ co; hit, _, _, hi, ho, _ = sc.ray_cast(dg, wp - cam_dir * 100.0, cam_dir)
        vis = bool(hit and ho is not None and ho.name == mesh.name and hi == fi)
        out.append([round(v.x * S, 2), round((1 - v.y) * S, 2)] + [round(g, 5) for g in gen_of(rc)] + [fi, FACE_PART[fi], int(vis)])
    ev.to_mesh_clear(); return out


def _proj_np(Pw):
    """Vectorised bpy_extras world_to_camera_view for the ORTHO camera (same formula), Pw: (N, 3) world coords -> (N, 3)."""
    import numpy as np
    Mi = np.array(cam.matrix_world.normalized().inverted()); fr = cam.data.view_frame(scene=sc)
    min_x, max_x, min_y, max_y = fr[2].x, fr[1].x, fr[1].y, fr[0].y
    L = Pw @ Mi[:3, :3].T + Mi[:3, 3]
    return np.stack([(L[:, 0] - min_x) / (max_x - min_x), (L[:, 1] - min_y) / (max_y - min_y), -L[:, 2]], 1)


def _mesh_np(me, mw):
    import numpy as np
    n = len(me.vertices); co = np.empty(n * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(n, 3).astype(np.float64)
    M = np.array(mw); W = co @ M[:3, :3].T + M[:3, 3]
    ls = np.empty(len(me.polygons), np.int64); lt = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get("loop_start", ls); me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), np.int64); me.loops.foreach_get("vertex_index", lv)
    return co, W, ls, lt, lv


def face_area_px_fast():
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get(); ev = mesh.evaluated_get(dg); me = ev.to_mesh()
    co, W, ls, lt, lv = _mesh_np(me, ev.matrix_world); P = _proj_np(W); X = P[:, 0] * S; Y = (1 - P[:, 1]) * S
    nxt = np.arange(len(lv)) + 1; end = ls + lt; nxt[end - 1] = ls                       # next loop index inside each polygon (wraps)
    cross = X[lv] * Y[lv[nxt]] - X[lv[nxt]] * Y[lv]; area = np.abs(np.add.reduceat(cross, ls)) / 2
    ev.to_mesh_clear(); return [round(float(a), 2) for a in area]


def surf_samples_fast(part_png):
    """surf_samples with whole-array maths. Visibility comes from the PART pass already rendered this frame (face index in G
    at the centroid pixel) for faces of >= 4 px projected area; smaller faces use the exact v2 camera-ray test."""
    import numpy as np
    dg = bpy.context.evaluated_depsgraph_get(); ev = mesh.evaluated_get(dg); me = ev.to_mesh(); mw = ev.matrix_world
    co, W, ls, lt, lv = _mesh_np(me, mw)
    nrm = np.empty(len(me.polygons) * 3, np.float32); me.polygons.foreach_get("normal", nrm); nrm = nrm.reshape(-1, 3).astype(np.float64)
    nmat = np.array(mw.to_3x3().inverted().transposed()); nw = nrm @ nmat.T; nw /= np.linalg.norm(nw, axis=1, keepdims=True).clip(1e-12)
    cam_dir = np.array((cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized())
    cnt = lt.astype(np.float64); cen = np.stack([np.add.reduceat(co[lv, k], ls) for k in range(3)], 1) / cnt[:, None]
    P = _proj_np(cen @ np.array(mw)[:3, :3].T + np.array(mw)[:3, 3]); px = P[:, 0] * S; py = (1 - P[:, 1]) * S
    rco = np.empty(len(mesh.data.vertices) * 3, np.float32); mesh.data.vertices.foreach_get("co", rco); rco = rco.reshape(-1, 3).astype(np.float64)
    rl = np.empty(len(mesh.data.loops), np.int64); mesh.data.loops.foreach_get("vertex_index", rl)
    rcen = np.stack([np.add.reduceat(rco[rl, k], ls) for k in range(3)], 1) / cnt[:, None]
    Lc, Sz = np.array(mesh.data.texspace_location), np.array(mesh.data.texspace_size); gen = (rcen - (Lc - Sz)) / (2 * Sz)
    area = np.array(face_area_px_fast())
    img = bpy.data.images.load(part_png, check_existing=False); img.colorspace_settings.name = "Non-Color"; w_, h_ = img.size; pix = np.empty(w_ * h_ * 4, np.float32); img.pixels.foreach_get(pix)   # raw values: no sRGB decode
    pix = pix.reshape(h_, w_, 4)[::-1]; bpy.data.images.remove(img); fidx = np.rint(pix[..., 1] * 65535).astype(np.int64); alpha = pix[..., 3]
    out = []
    for fi in SURF_FACES:
        if nw[fi] @ cam_dir > -0.3: continue
        if area[fi] >= 4:
            xi, yi = int(np.floor(px[fi])), int(np.floor(py[fi]))
            vis = int(0 <= xi < w_ and 0 <= yi < h_ and alpha[yi, xi] > 0 and fidx[yi, xi] == fi)
        else:                                                                   # small face: exact v2 camera-ray test (vis stays 0/1; surf_pilot_check reads it as a bool)
            wp = mw @ Vector(cen[fi]); hit, _, _, hi, ho, _ = sc.ray_cast(dg, wp - Vector(cam_dir) * 100.0, Vector(cam_dir))
            vis = int(bool(hit and ho is not None and ho.name == mesh.name and hi == fi))
        out.append([round(float(px[fi]), 2), round(float(py[fi]), 2)] + [round(float(g), 5) for g in gen[fi]] + [fi, FACE_PART[fi], vis])
    ev.to_mesh_clear(); return out, [round(float(a), 2) for a in area]


def face_area_px():
    """v3 (Astra 1561397810287349761): projected screen area in pixels of EVERY face this frame (shoelace over the
    deformed, camera-projected vertices), indexed by face; the checker looks it up per pixel through the face index."""
    dg = bpy.context.evaluated_depsgraph_get(); ev = mesh.evaluated_get(dg); me = ev.to_mesh(); mw = ev.matrix_world
    P = [world_to_camera_view(sc, cam, mw @ v.co) for v in me.vertices]; out = []
    for p_ in me.polygons:
        q = [(P[i].x * S, (1 - P[i].y) * S) for i in p_.vertices]
        out.append(round(abs(sum(q[k][0] * q[(k + 1) % len(q)][1] - q[(k + 1) % len(q)][0] * q[k][1] for k in range(len(q)))) / 2, 2))
    ev.to_mesh_clear(); return out


SURF_FACES = list(range(len(mesh.data.polygons))) if B.get("surface_pass") else []   # v4 (stage-1 count fix): every face is a sample face
if B.get("surface_pass"):
    os.makedirs(OUT, exist_ok=True); build_parts(); surf_points(os.path.join(OUT, "surf_points.npy"))
    json.dump({"texspace_location": list(mesh.data.texspace_location), "texspace_size": list(mesh.data.texspace_size), "polygons": len(mesh.data.polygons),
               "part_names": PART_NAMES, "faces_per_part": {PART_NAMES[k]: FACE_PART.count(k) for k in range(len(PART_NAMES))}}, open(os.path.join(OUT, "surf_texspace.json"), "w"))


for ch in B["characters"]:
    cdir = os.path.join(OUT, ch["name"]); os.makedirs(cdir, exist_ok=True)
    pb["Head"].scale = Vector([ch.get("head_scale", 1.0)] * 3)
    ROOT.scale = Vector(ch.get("body_scale", [1.0, 1.0, 1.0]))                  # T1b proportions (opt-in; default identity)
    if STRIPE:
        st = ch.get("stripe"); lo = STRIPE["z0"] + (STRIPE["z1"] - STRIPE["z0"]) * (st["from"] if st else 0); hi = STRIPE["z0"] + (STRIPE["z1"] - STRIPE["z0"]) * (st["to"] if st else 0)
        nband = 0
        for p in STRIPE["shirt"]:
            band = bool(st) and lo <= p.center.z <= hi; p.material_index = STRIPE["idx"] if band else STRIPE["orig"][p.index]; nband += band
        print("STRIPE_FACES", ch["name"], nband, "of", len(STRIPE["shirt"]), flush=True)
    ROOT.location.z = 0.0
    ROOT.rotation_euler.z = yaw_for(1); apply(*capture("Idle", 0)); pb["Head"].scale = Vector([ch.get("head_scale", 1.0)] * 3); legs(ch); bpy.context.view_layer.update()
    up = (pb["Head"].head - pb["Hips"].head).normalized(); arm["idle_foot_low"] = min(pb[JOINTS["r_ankle"]].head.dot(up), pb[JOINTS["l_ankle"]].head.dot(up))
    set_look(ch); render(os.path.join(cdir, "ref.png")); set_look(ch, True); render(os.path.join(cdir, "ref_labels.png"))
    if B.get("surface_pass"): surf_render(os.path.join(cdir, "ref_surf.png")); part_render(os.path.join(cdir, "ref_part.png"))
    rp, rw = joints(); json.dump({"pose": "idle frame 0, facing right", "joints_px": rp, "joints_world": rw, **proportions(ch), **({"face_area_px": face_area_px()} if B.get("surface_pass") else {})}, open(os.path.join(cdir, "ref.json"), "w"))
    for cl in ch["clips"]:
        d = os.path.join(cdir, cl["name"]); os.makedirs(os.path.join(d, "rgba"), exist_ok=True); os.makedirs(os.path.join(d, "labels"), exist_ok=True)
        recs = []; t = 0; prev = None; yaw = yaw_for(cl["segments"][0]["facing"]); clocks = {}
        for si, seg in enumerate(cl["segments"]):
            for u in range(seg["frames"]):
                base = BASE_OF[seg["action"]]; clocks[base] = (clocks.get(base, cl.get("phase0", 0.0) * 25) + 1) if base not in MOCAP else u
                cur = capture(base, clocks[base])
                w = min(1.0, (u + 1) / BLEND) if (prev is not None and u < BLEND) else 1.0
                pose = mix(prev, cur, w) if w < 1.0 else cur
                apply(*pose); pb["Head"].scale = Vector([ch.get("head_scale", 1.0)] * 3); legs(ch); bpy.context.view_layer.update()
                if seg["action"] == "Squat":
                    AC.squat(arm, bpy.context, math.sin(math.pi * u / max(1, seg["frames"] - 1)) ** 2)
                elif seg["action"] == "Wave" and not B.get("wave_bvh"):
                    (AC.wave2 if B.get("wave", "wave2") == "wave2" else AC.wave)(arm, bpy.context, min(1.0, u / 8.0), 2 * math.pi * 2.0 * u / FPS)
                f_ = seg["facing"]
                if isinstance(f_, dict) and "yaw_to" in f_:                         # MULTIANGLE: exact linear sweep over the segment (continuous turn)
                    yaw = B.get("right_yaw", math.pi) + math.radians(f_["yaw_from"] + (f_["yaw_to"] - f_["yaw_from"]) * u / max(1, seg["frames"] - 1)); ROOT.rotation_euler.z = yaw
                else:
                    target = yaw_for(f_); yaw += max(-math.pi / TURN, min(math.pi / TURN, target - yaw)); ROOT.rotation_euler.z = yaw
                bpy.context.view_layer.update()
                set_look(ch); timed("rgba", render, os.path.join(d, "rgba", f"{t:05d}.png")); set_look(ch, True); timed("labels", render, os.path.join(d, "labels", f"{t:05d}.png"), True)
                if B.get("surface_pass"):
                    os.makedirs(os.path.join(d, "surf"), exist_ok=True); timed("surf", surf_render, os.path.join(d, "surf", f"{t:05d}.png"))
                    os.makedirs(os.path.join(d, "part"), exist_ok=True); timed("part", part_render, os.path.join(d, "part", f"{t:05d}.png"))
                jp, jw = joints(); recs.append({"t": t, "joints_px": jp, "joints_world": jw, "segment": si, "action": seg["action"],
                                                "facing": seg["facing"], "yaw": round(yaw, 4), "blend_w": round(w, 3), **proportions(ch), **((dict(zip(("surf_samples", "face_area_px"), surf_samples_fast(os.path.join(d, "part", f"{t:05d}.png")))) if FAST else {"surf_samples": surf_samples(), "face_area_px": face_area_px()}) if B.get("surface_pass") else {}),
                                                **({"shirt_track": track_px(), "shirt_track_vertices": TRACK, "shirt_track_faces": TRACK_FACES, "pattern_axes": PAT_AXES} if TRACK else {})})
                t += 1
            prev = (arm.matrix_basis.copy(), {b.name: (b.location.copy(), b.rotation_quaternion.copy(), b.scale.copy()) for b in pb})   # last shown pose (incl. extras)
        json.dump({"schema": "ff_cartoon_blender_v2", "fps": FPS, "character": ch, "clip": cl, "frames": recs}, open(os.path.join(d, "gt.json"), "w"))
    print("CHAR_OK", ch["name"], flush=True)
if B.get("surface_pass") or B.get("timing"):                                       # opt-in only: default runs write exactly the same files as before
    json.dump({k: {"n": len(v), "mean_s": round(sum(v) / max(1, len(v)), 4)} for k, v in TIMES.items()}, open(os.path.join(OUT, "_timing_%s.json" % os.path.basename(sys.argv[-1]).split(".")[0]), "w"))
print("BATCH_OK", len(B["characters"]))

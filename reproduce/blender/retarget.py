# Retarget CMU motion capture (BVH, cgspeed conversion; CMU: no restrictions on use) onto the CC0 Quaternius human
# (Claude 2026-10-17; Payton 1560885169148264471 "fix the waving animation": real captured motion, not hand-keyed poses).
# Method: per frame, every mapped target bone is rotated (armature space, parent -> child order) so its direction matches the
# source bone's world direction, after a fixed yaw that aligns the source's initial hip axis with the target's. Hips height is
# scaled by the leg-length ratio; horizontal root motion is dropped (the character stays centred; the street scrolls).
import math
import bpy
from mathutils import Matrix, Vector, Quaternion

CHAIN = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
         "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
         "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
# source name aliases (cgspeed CMU BVH)
ALIAS = {"Spine2": ["Spine2", "Spine1", "Spine"], "Spine1": ["Spine1", "Spine"], "Neck": ["Neck1", "Neck"], "LeftShoulder": ["LeftShoulder"],
         "RightShoulder": ["RightShoulder"]}
CHILD = {"Hips": "Spine", "Spine": "Spine1", "Spine1": "Spine2", "Spine2": "Neck", "Neck": "Head", "Head": None,
         "LeftShoulder": "LeftArm", "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand", "LeftHand": None,
         "RightShoulder": "RightArm", "RightArm": "RightForeArm", "RightForeArm": "RightHand", "RightHand": None,
         "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": None, "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": None}


class Retarget:
    def __init__(s, tgt_arm, bvh_path, ctx, extra_yaw=0.0):
        s.t, s.ctx = tgt_arm, ctx
        before = set(bpy.data.objects)
        bpy.ops.import_anim.bvh(filepath=bvh_path, global_scale=1.0, rotate_mode="NATIVE", axis_forward="-Z", axis_up="Y")
        s.src = next(o for o in bpy.data.objects if o not in before and o.type == "ARMATURE")
        s.src.hide_render = True
        a = s.src.animation_data.action; s.f0, s.f1 = int(a.frame_range[0]), int(a.frame_range[1])
        s.names = {}
        for b in CHAIN:
            for n in ALIAS.get(b, [b]):
                if n in s.src.pose.bones:
                    s.names[b] = n; break
        sp = s.src.pose.bones
        s.ctx.scene.frame_set(s.f0)
        w = s.src.matrix_world
        lat_s = (w @ sp[s.names["RightUpLeg"]].head - w @ sp[s.names["LeftUpLeg"]].head); lat_s.z = 0
        tp = s.t.pose.bones; tw = s.t.matrix_world
        lat_t = (tw @ tp["RightUpLeg"].head - tw @ tp["LeftUpLeg"].head); lat_t.z = 0
        d = math.atan2(lat_t.y, lat_t.x) - math.atan2(lat_s.y, lat_s.x)          # yaw ONLY: rotation_difference of near-opposite
        s.align = Quaternion((0, 0, 1), d + extra_yaw)                                        # vectors can pick a horizontal axis (front/back flip)
        legs = lambda P, W, n: (W @ P[n("LeftUpLeg")].head - W @ P[n("LeftFoot")].head).length
        s.ratio = legs(tp, tw, lambda x: x) / max(1e-6, legs(sp, w, lambda x: s.names[x]))
        s.src_hip0 = (w @ sp[s.names["Hips"]].head).z; s.tgt_hip0 = (tw @ tp["Hips"].head).z

    def frames(s, fps=24):
        """Source frame numbers resampled from the BVH rate to fps."""
        rate = round(1.0 / bpy.context.scene.render.fps_base / max(1, bpy.context.scene.render.fps)) if False else None
        step = max(1, int(round(120 / fps)))
        return list(range(s.f0, s.f1 + 1, step))

    def src_dir(s, b, child):
        sp = s.src.pose.bones; w = s.src.matrix_world
        h = w @ sp[s.names[b]].head
        tail = (w @ sp[s.names[child]].head) if (child and child in s.names) else (w @ sp[s.names[b]].tail)
        return (s.align @ (tail - h)).normalized()

    def apply(s, frame):
        """Pose the target at a source frame (target action must be detached)."""
        s.ctx.scene.frame_set(frame); tp = s.t.pose.bones; tw = s.t.matrix_world; twi = tw.inverted()
        for b in tp:
            b.rotation_mode = "QUATERNION"; b.rotation_quaternion = Quaternion(); b.location = Vector()
        s.ctx.view_layer.update()
        sp = s.src.pose.bones; w = s.src.matrix_world
        dz = ((w @ sp[s.names["Hips"]].head).z - s.src_hip0) * s.ratio                       # hips height change only
        tp["Hips"].matrix = Matrix.Translation(twi.to_3x3() @ Vector((0, 0, dz))) @ tp["Hips"].matrix; s.ctx.view_layer.update()
        # hips orientation from the source hip axis and spine
        for b in CHAIN:
            if b not in s.names: continue
            child = CHILD[b]; want = s.src_dir(b, child)
            hb = tw @ tp[b].head; cur_tail = (tw @ tp[child].head) if child else (tw @ tp[b].tail)
            have = (cur_tail - hb).normalized()
            if have.length < 1e-6 or want.length < 1e-6: continue
            q = have.rotation_difference(want)                                                 # world rotation
            ax, ang = q.to_axis_angle()
            ax_arm = (twi.to_3x3() @ ax).normalized(); piv = tp[b].head.copy()
            tp[b].matrix = Matrix.Translation(piv) @ Matrix.Rotation(ang, 4, ax_arm) @ Matrix.Translation(-piv) @ tp[b].matrix
            s.ctx.view_layer.update()
            if b == "Hips":                                                                    # also match the hip (lateral) axis
                want_l = (s.align @ ((w @ sp[s.names["RightUpLeg"]].head) - (w @ sp[s.names["LeftUpLeg"]].head))).normalized()
                have_l = ((tw @ tp["RightUpLeg"].head) - (tw @ tp["LeftUpLeg"].head)).normalized()
                up = (tw @ tp["Spine"].head - tw @ tp["Hips"].head).normalized()
                pl = lambda v: (v - up * v.dot(up)).normalized()
                q2 = pl(have_l).rotation_difference(pl(want_l)); ax2, an2 = q2.to_axis_angle(); ax2 = (twi.to_3x3() @ ax2).normalized()
                piv = tp["Hips"].head.copy()
                tp["Hips"].matrix = Matrix.Translation(piv) @ Matrix.Rotation(an2, 4, ax2) @ Matrix.Translation(-piv) @ tp["Hips"].matrix
                s.ctx.view_layer.update()

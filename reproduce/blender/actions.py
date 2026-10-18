# Authored actions for the CC0 human (Claude 2026-10-17; Payton 1560882287510560859: add squat and "stop + wave to camera").
# The asset has Walk/Run/Idle/Jump but no squat or wave. These are posed procedurally on top of the Idle frame, with rotations
# about the character's own anatomical axes (lateral = left hip -> right hip, forward = up x lateral), applied in armature
# space so they do not depend on the FBX bone roll. Call after frame_set(Idle) with the action detached.
import math
from mathutils import Matrix, Vector

SIGN = {"flex": 1.0, "abduct": -1.0}         # abduct -1: arm goes out to the side and up (axis test 2026-10-17, runs/wave_axes.png)


def _axes(arm):
    pb = arm.pose.bones
    lat = (pb["RightUpLeg"].head - pb["LeftUpLeg"].head).normalized()                 # armature space
    up = (pb["Head"].head - pb["Hips"].head).normalized()
    fwd = up.cross(lat).normalized()
    return lat, up, fwd


def rot_about(pb, axis, ang, pivot=None):
    """Rotate pose bone pb by ang (rad) about an armature-space axis through pivot (default its head)."""
    pivot = pb.head.copy() if pivot is None else pivot
    R = Matrix.Translation(pivot) @ Matrix.Rotation(ang, 4, axis) @ Matrix.Translation(-pivot)
    pb.matrix = R @ pb.matrix


def squat(arm, ctx, s):
    """s in [0,1]: hips sink with knees forward, feet stay planted, torso leans a little forward."""
    lat, up, fwd = _axes(arm); pb = arm.pose.bones; th = math.radians(50) * s
    L = (pb["RightLeg"].head - pb["RightUpLeg"].head).length + (pb["RightFoot"].head - pb["RightLeg"].head).length
    drop = L * (1 - math.cos(th))
    pb["Hips"].matrix = Matrix.Translation(-up * drop) @ pb["Hips"].matrix; ctx.view_layer.update()
    rot_about(pb["Spine"], lat, SIGN["flex"] * math.radians(22) * s); ctx.view_layer.update()
    for side in ("Right", "Left"):
        rot_about(pb[f"{side}UpLeg"], lat, SIGN["flex"] * th); ctx.view_layer.update()
        rot_about(pb[f"{side}Leg"], lat, -SIGN["flex"] * 2 * th); ctx.view_layer.update()
        rot_about(pb[f"{side}Foot"], lat, SIGN["flex"] * th); ctx.view_layer.update()
    for side in ("Right", "Left"):                                                         # arms forward for balance
        rot_about(pb[f"{side}Arm"], lat, SIGN["flex"] * math.radians(70) * s); ctx.view_layer.update()
    ground(arm, ctx)


def ground(arm, ctx, z0=None):
    """Keep the lowest foot on the floor: shift the hips so min(foot z) returns to the idle value (armature space)."""
    pb = arm.pose.bones; up = (pb["Head"].head - pb["Hips"].head).normalized()
    low = min(pb["RightFoot"].head.dot(up), pb["LeftFoot"].head.dot(up))
    target = arm.get("idle_foot_low", low) if z0 is None else z0
    pb["Hips"].matrix = Matrix.Translation(up * (target - low)) @ pb["Hips"].matrix; ctx.view_layer.update()


def wave(arm, ctx, raise_s, ph):
    """Right arm raised out to the side (raise_s in [0,1]) and the forearm waving (phase ph)."""
    import os
    lat, up, fwd = _axes(arm); pb = arm.pose.bones
    ax = {"fwd": fwd, "lat": lat}[os.environ.get("WAVE_AXIS", "fwd")]; sg = float(os.environ.get("WAVE_SIGN", SIGN["abduct"]))
    rot_about(pb["RightArm"], ax, sg * math.radians(150) * raise_s); ctx.view_layer.update()
    rot_about(pb["RightForeArm"], ax, sg * math.radians(25) * math.sin(ph) * raise_s); ctx.view_layer.update()


def wave2(arm, ctx, raise_s, ph):
    """Natural wave (Payton 1560883911637209193): upper arm out to ~shoulder height, elbow bent ~90 deg so the forearm points
    up, the forearm + hand swinging side to side from the elbow, a slight head tilt and shoulder lift."""
    lat, up, fwd = _axes(arm); pb = arm.pose.bones; sg = SIGN["abduct"]
    rot_about(pb["RightShoulder"], fwd, sg * math.radians(10) * raise_s); ctx.view_layer.update()
    rot_about(pb["RightArm"], fwd, sg * math.radians(80) * raise_s); ctx.view_layer.update()
    rot_about(pb["RightForeArm"], fwd, sg * math.radians(85 + 22 * math.sin(ph)) * raise_s); ctx.view_layer.update()
    rot_about(pb["RightHand"], fwd, sg * math.radians(10 * math.sin(ph + .6)) * raise_s); ctx.view_layer.update()
    rot_about(pb["Head"], fwd, -sg * math.radians(6) * raise_s); ctx.view_layer.update()

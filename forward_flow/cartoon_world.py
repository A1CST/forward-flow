"""Headless, deterministic labeled renderer for Forward-Flow Cartoon.

It intentionally draws a small original 2D cartoon domain rather than copying
an external style.  The output separates environment and character layers so
the two learned renderers can be trained and tested independently.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FPS = 16
SIZE = 256
JOINTS = ("head", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist", "pelvis", "r_knee", "r_ankle", "l_knee", "l_ankle")
BONES = ((0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (8, 11), (11, 12))


@dataclass(frozen=True)
class Character:
    name: str
    skin: tuple[int, int, int]
    hair: tuple[int, int, int]
    shirt: tuple[int, int, int]
    pants: tuple[int, int, int]
    accent: tuple[int, int, int]
    scale: float


ROSTER = (
    Character("mira", (250, 202, 166), (42, 31, 52), (80, 140, 224), (47, 65, 103), (249, 201, 74), 1.00),
    Character("sol", (137, 88, 64), (34, 26, 24), (232, 107, 92), (46, 74, 68), (242, 224, 159), 1.05),
    Character("ivy", (224, 171, 132), (74, 47, 40), (104, 178, 132), (60, 83, 122), (246, 160, 87), 0.94),
)


def pose(action: str, phase: float, x: float, facing: int) -> np.ndarray:
    """Thirteen local-to-world joints; y is up and the feet contact y=0."""
    run = action == "run"
    stride = 0.55 if run else 0.33
    arm = 0.72 if run else 0.46
    if action == "idle":
        stride, arm = 0.03, 0.08
    q = math.sin(phase)
    rel = np.zeros((13, 2), np.float32)
    rel[8] = (0, 1.05)
    rel[1] = (0.06 * facing, 1.85)
    rel[0] = (0.09 * facing, 2.18)
    rel[2], rel[5] = (0.26 * facing, 1.80), (-0.26 * facing, 1.80)
    for shoulder, elbow, wrist, sign in ((2, 3, 4, 1), (5, 6, 7, -1)):
        a = sign * arm * q
        rel[elbow] = rel[shoulder] + (0.20 * facing + 0.14 * a, -0.42)
        rel[wrist] = rel[elbow] + (0.14 * facing + 0.10 * a, -0.36)
    for knee, ankle, sign in ((9, 10, 1), (11, 12, -1)):
        a = sign * stride * q
        rel[knee] = (0.22 * facing + a, 0.53)
        rel[ankle] = (0.38 * facing + 0.55 * a, 0.0)
    if action.startswith("guard"):
        rel[3], rel[4] = (0.16 * facing, 1.72), (0.10 * facing, 2.02)
        rel[6], rel[7] = (-0.02 * facing, 1.70), (0.04 * facing, 2.00)
    elif action.startswith("lunge"):
        rel[8] += (0.24 * facing, -0.12); rel[10] += (0.42 * facing, 0); rel[12] -= (0.28 * facing, 0)
    elif action.startswith("dodge"):
        rel[:, 0] -= 0.26 * facing; rel[:, 1] -= 0.10
    elif action.startswith("exit"):
        rel[:, 0] += 0.22 * facing
    return rel + np.array((x, 0), np.float32)


def project(world: np.ndarray, camera_x: float, size: int = SIZE, zoom: float = 66.0) -> np.ndarray:
    return np.column_stack((size * .5 + (world[:, 0] - camera_x) * zoom, size * .78 - world[:, 1] * zoom))


def _rgba(color, alpha=255): return tuple(color) + (alpha,)


def draw_environment(size: int, camera_x: float, scene_seed: int) -> tuple[Image.Image, list[dict]]:
    rng = np.random.default_rng(scene_seed)
    im = Image.new("RGBA", (size, size), _rgba((152, 203, 236)))
    d = ImageDraw.Draw(im)
    d.ellipse((-30, 20, 120, 125), fill=_rgba((188, 224, 243)))
    d.ellipse((100, 18, 290, 145), fill=_rgba((177, 215, 239)))
    ground = int(size * .78)
    d.rectangle((0, ground, size, size), fill=_rgba((115, 176, 119)))
    d.rectangle((0, ground, size, ground + 8), fill=_rgba((78, 130, 92)))
    entities=[]
    for i, kind in enumerate(("building", "bush", "lamp", "tree", "bench", "bush", "building")):
        x = -4.2 + i * 1.45 + float(rng.uniform(-.18, .18)); w = float(rng.uniform(.45, .85)); h = float(rng.uniform(.55, 1.45))
        px = int(size*.5 + (x-camera_x)*66); py = ground
        ent={"id":f"{kind}_{i}","kind":kind,"x":round(x,3),"y":0,"z":1 if kind in ("bush","bench") else 2,"w":round(w,3),"h":round(h,3)}
        entities.append(ent)
        if kind == "building":
            box=(px-int(w*34), py-int(h*66), px+int(w*34), py)
            d.rounded_rectangle(box, radius=8, fill=_rgba((228, 180, 118)), outline=_rgba((106, 86, 88)), width=3)
            for wx in range(box[0]+10, box[2]-6, 17):
                d.rectangle((wx, box[1]+15, wx+8, box[1]+28), fill=_rgba((121, 181, 205)))
        elif kind == "bush":
            c=(58, 126, 85)
            for j in range(3): d.ellipse((px-int(w*45)+j*int(w*25), py-int(h*42)-8, px-int(w*45)+(j+2)*int(w*25), py+5), fill=_rgba(c))
        elif kind == "tree":
            d.rectangle((px-5, py-int(h*52), px+5, py), fill=_rgba((103, 74, 55)))
            d.ellipse((px-int(w*45), py-int(h*82), px+int(w*45), py-int(h*25)), fill=_rgba((60, 135, 88)))
        elif kind == "lamp":
            d.rectangle((px-3, py-int(h*70), px+3, py), fill=_rgba((60, 74, 91)))
            d.ellipse((px-10, py-int(h*74), px+10, py-int(h*54)), fill=_rgba((255, 222, 126)))
        else:
            d.rectangle((px-int(w*40), py-int(h*35), px+int(w*40), py-int(h*25)), fill=_rgba((117, 79, 58)))
            d.rectangle((px-int(w*30), py-int(h*25), px-int(w*23), py), fill=_rgba((89, 61, 50)))
            d.rectangle((px+int(w*23), py-int(h*25), px+int(w*30), py), fill=_rgba((89, 61, 50)))
    return im, entities


PARTS = {"head":1,"torso":2,"r_upper_arm":3,"r_lower_arm":4,"l_upper_arm":5,"l_lower_arm":6,"r_upper_leg":7,"r_lower_leg":8,"l_upper_leg":9,"l_lower_leg":10,"hands":11,"feet":12}
BONE_PART = {(0,1):1,(1,2):2,(2,3):3,(3,4):4,(1,5):2,(5,6):5,(6,7):6,(1,8):2,(8,9):7,(9,10):8,(8,11):9,(11,12):10}
FAR_PARTS = {5,6,9,10}

def draw_character(size: int, px: np.ndarray, char: Character):
    """Return composite RGBA, alpha mask, far RGBA, near RGBA, and 8-bit part labels."""
    far=Image.new("RGBA",(size,size)); near=Image.new("RGBA",(size,size)); labels=Image.new("L",(size,size)); fm=Image.new("L",(size,size)); nm=Image.new("L",(size,size))
    fd,nd,ld=ImageDraw.Draw(far),ImageDraw.Draw(near),ImageDraw.Draw(labels); fmd,nmd=ImageDraw.Draw(fm),ImageDraw.Draw(nm); p=np.rint(px).astype(int); outline=(42,45,60)
    def limb(a,b,col,part):
        draw,alpha=(fd,fmd) if part in FAR_PARTS else (nd,nmd)
        draw.line((tuple(a),tuple(b)),fill=_rgba(outline),width=11,joint="curve"); draw.line((tuple(a),tuple(b)),fill=_rgba(col),width=7,joint="curve"); alpha.line((tuple(a),tuple(b)),fill=255,width=11); ld.line((tuple(a),tuple(b)),fill=part,width=7)
    for a,b in BONES:
        part=BONE_PART[(a,b)]; limb(p[a],p[b],char.pants if part>=7 else char.shirt,part)
    torso=(p[1]+p[8])//2; box=(torso[0]-14,torso[1]-18,torso[0]+14,torso[1]+23); nd.rounded_rectangle(box,radius=9,fill=_rgba(char.shirt),outline=_rgba(outline),width=3); nmd.rounded_rectangle(box,radius=9,fill=255); ld.rounded_rectangle(box,radius=9,fill=PARTS["torso"])
    head=(p[0,0]-15,p[0,1]-17,p[0,0]+15,p[0,1]+15); nd.ellipse(head,fill=_rgba(char.skin),outline=_rgba(outline),width=3); nd.arc((head[0],head[1]-1,head[2],head[1]+25),185,355,fill=_rgba(char.hair),width=7); nmd.ellipse(head,fill=255); ld.ellipse(head,fill=PARTS["head"])
    for a in (4,7):
        b=(p[a,0]-5,p[a,1]-4,p[a,0]+5,p[a,1]+4); nd.ellipse(b,fill=_rgba(char.accent)); nmd.ellipse(b,fill=255); ld.ellipse(b,fill=PARTS["hands"])
    for a in (10,12):
        b=(p[a,0]-6,p[a,1]-4,p[a,0]+6,p[a,1]+4); nd.ellipse(b,fill=_rgba(char.accent)); nmd.ellipse(b,fill=255); ld.ellipse(b,fill=PARTS["feet"])
    mask=Image.fromarray(np.maximum(np.asarray(fm),np.asarray(nm)).astype(np.uint8),"L")
    return Image.alpha_composite(far,near),mask,far,near,labels


def render_clip(out: Path, character: Character, action: str, frames: int, scene_seed: int, clip_seed: int, size: int=SIZE, split: str="fit") -> dict:
    out.mkdir(parents=True, exist_ok=False); (out/"frames").mkdir(); (out/"environment").mkdir(); (out/"character").mkdir(); (out/"character_far").mkdir(); (out/"character_near").mkdir(); (out/"part_labels").mkdir(); (out/"masks").mkdir()
    rng=np.random.default_rng(clip_seed); facing=1 if rng.random()>.5 else -1; phase=float(rng.uniform(0,2*math.pi)); x=-1.3*facing; speed=1.25 if action=="run" else .7 if action=="walk" else 0.0
    records=[]; entities=None
    for t in range(frames):
        if t and action in ("walk","run"): x += facing*speed/FPS
        camera_x=x*.72 + .18*math.sin(t/13)  # continuous follow camera, never a cut
        world=pose(action, phase, x, facing); px=project(world,camera_x,size)
        env,entities=draw_environment(size,camera_x,scene_seed); char,mask,far,near,labels=draw_character(size,px,character); frame=Image.alpha_composite(env,char).convert("RGB")
        frame.save(out/"frames"/f"{t:05d}.png"); env.convert("RGB").save(out/"environment"/f"{t:05d}.png"); char.save(out/"character"/f"{t:05d}.png"); far.save(out/"character_far"/f"{t:05d}.png"); near.save(out/"character_near"/f"{t:05d}.png"); labels.save(out/"part_labels"/f"{t:05d}.png"); mask.save(out/"masks"/f"{t:05d}.png")
        records.append({"t":t,"action":action,"phase":round(float(phase%(2*math.pi)),6),"joints_world":np.round(world,4).tolist(),"joints_px":np.round(px,3).tolist(),"camera":{"x":round(camera_x,4),"zoom":66.0},"depth_order":["environment","character_far","character_near"],"cut":False,"visible":bool(((px[:,0]>=0)&(px[:,0]<size)&(px[:,1]>=0)&(px[:,1]<size)).all())})
        phase += 2*math.pi*(1.65 if action=="run" else 1.0 if action=="walk" else .35)/FPS
    gt={"schema":"ff_cartoon_v1","split":split,"fps":FPS,"size":[size,size],"character":asdict(character),"action":action,"scene_seed":scene_seed,"clip_seed":clip_seed,"joint_names":JOINTS,"bones":BONES,"entities":entities,"frames":records}
    (out/"gt.json").write_text(json.dumps(gt,indent=1)); return gt


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--clips",type=int,default=12); p.add_argument("--frames",type=int,default=64); p.add_argument("--size",type=int,default=SIZE); a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); index=[]
    actions=("idle","walk","run","guard_left","guard_right","lunge_left","lunge_right","dodge_left","dodge_right","exit_left","exit_right")
    for i in range(a.clips):
        c=ROSTER[i%len(ROSTER)]; act=actions[i%len(actions)]; split="heldout" if c.name=="ivy" else "fit"; clip=f"{c.name}_{act}_{i:04d}"; gt=render_clip(a.out/clip,c,act,a.frames,100+i,1000+i,a.size,split); index.append({"clip":clip,"character":c.name,"split":split,"action":act,"frames":a.frames,"all_visible":all(x["visible"] for x in gt["frames"]),"cuts":0})
    (a.out/"index.json").write_text(json.dumps({"schema":"ff_cartoon_v1","clips":index},indent=1)); print(json.dumps({"clips":len(index),"frames":len(index)*a.frames,"out":str(a.out)}))


if __name__ == "__main__": main()

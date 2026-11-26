// Coordinates throughout physics: calibrated world X,Y,Z (metres).
const defaultDoorStates = {door_room_far:false, door_room_near:false, door_c5_end:true, door_corridor_end:true};
export function defaultClosedDoors(site) {
  return new Set(site.objects.filter(o => o.role === 'door' &&
    (Object.hasOwn(defaultDoorStates, o.id) ? defaultDoorStates[o.id] : o.occludes !== false)).map(o => o.id));
}

export function obstacles(site, closed = new Set()) {
  return site.objects.flatMap(o => {
    if (o.occludes === false && !closed.has(o.id)) return [];
    if (o.kind === 'pillar') return [{kind:'cylinder', center:[...o.center,0], radius:o.radius, height:o.height, id:o.id}];
    return (o.points || []).slice(1).map((b,i) => {
      const a=o.points[i], dx=b[0]-a[0], dy=b[1]-a[1], length=Math.hypot(dx,dy);
      return {kind:'box', center:[(a[0]+b[0])/2,(a[1]+b[1])/2,0], axis:[dx/length,dy/length], half:[length/2,o.width/2],height:o.height,id:o.id};
    });
  });
}
export function intersection(origin, target, o) {
  const d=target.map((v,i)=>v-origin[i]), p=origin.map((v,i)=>v-o.center[i]);
  let lo=0, hi=1;
  function slab(a,b,h0,h1){if(Math.abs(b)<1e-10)return a>=h0&&a<=h1;let t0=(h0-a)/b,t1=(h1-a)/b;lo=Math.max(lo,Math.min(t0,t1));hi=Math.min(hi,Math.max(t0,t1));return lo<=hi;}
  if(!slab(p[2],d[2],0,o.height))return null;
  if(o.kind==='box'){
    const [x,y]=o.axis;
    if(!slab(p[0]*x+p[1]*y,d[0]*x+d[1]*y,-o.half[0],o.half[0])||!slab(-p[0]*y+p[1]*x,-d[0]*y+d[1]*x,-o.half[1],o.half[1]))return null;
  }else{
    const a=d[0]**2+d[1]**2,b=2*(p[0]*d[0]+p[1]*d[1]),c=p[0]**2+p[1]**2-o.radius**2;
    if(a<1e-12){if(c>0)return null;}else{const disc=b*b-4*a*c;if(disc<0)return null;lo=Math.max(lo,(-b-Math.sqrt(disc))/(2*a));hi=Math.min(hi,(-b+Math.sqrt(disc))/(2*a));}
  }
  return lo<=hi&&hi>1e-5&&lo<1-1e-5?Math.max(0,lo):null;
}
export function firstHit(origin,target,objects){let t=1,hit=null;for(const o of objects){const u=intersection(origin,target,o);if(u!==null&&u<t){t=u;hit=o;}}return hit?{t,object:hit,point:origin.map((v,i)=>v+(target[i]-v)*t)}:null;}
function lensDistortion(x, y, [k1, k2, p1, p2, k3]) {
  const r2 = x*x + y*y;
  return [1 + r2*(k1 + r2*(k2 + r2*k3)),
    2*p1*x*y + p2*(r2 + 2*x*x), p1*(r2 + 2*y*y) + 2*p2*x*y];
}

function undistortedRay(u, v, distortion) {
  let x = u, y = v;
  if (distortion) {
    for (let iteration = 0; iteration < 15; iteration++) {
      const [radial, tx, ty] = lensDistortion(x, y, distortion);
      const nextX = (u - tx) / radial, nextY = (v - ty) / radial;
      if (Math.abs(nextX - x) + Math.abs(nextY - y) < 1e-8) break;
      x = nextX;
      y = nextY;
    }
  }
  return [x, y];
}

export function project(cam, point) {
  const world = point.map((value, i) => value - cam.center[i]);
  const [x, y, z] = cam.R.map(axis => axis.reduce((sum, value, i) => sum + value * world[i], 0));
  const [[fx, , cx], [, fy, cy]] = cam.K;
  const depth = (cam.depth_sign ?? 1)*z;
  // Outside the calibrated lens domain a distortion polynomial can fold rays back into the image.
  const validRay = depth > (cam.min_depth ?? .01) &&
    (cam.max_ray === undefined || Math.hypot(x, y) <= cam.max_ray * depth);
  let u = x/z, v = y/z;
  if (cam.dist) {
    const [radial, tx, ty] = lensDistortion(u, v, cam.dist);
    u = u*radial + tx;
    v = v*radial + ty;
  }
  return {uv:[fx*u + cx, fy*v + cy], depth, validRay};
}
export function inFrame(cam,p){const {uv,validRay}=project(cam,p);return validRay&&uv[0]>=0&&uv[1]>=0&&uv[0]<cam.size[0]&&uv[1]<cam.size[1];}
export function groundVisible(cam,point,occluders){return inFrame(cam,point)&&!firstHit(cam.center,point,occluders);}

export function bodyVisible(cam, {x, y}, box, solids) {
  const body = {kind: 'cylinder', center: [x, y, 0], radius: 0.25, height: 1.8};
  const [left, top, right, bottom] = box;
  const [[fx, , cx], [, fy, cy]] = cam.K;
  const origin = cam.center;
  const reach = (cam.depth_sign ?? 1) *
    (Math.hypot(x - origin[0], y - origin[1], body.height / 2 - origin[2]) + body.height);
  const [rightAxis, downAxis, forwardAxis] = cam.R;
  const steps = 12, minVisible = 0.15;
  let hits = 0, visible = 0;
  // Sample equal screen areas and compare obstacle depth to the first body surface.
  for (let row = 0; row < steps; row++) {
    const v = (top + (row + 0.5) * (bottom - top) / steps - cy) / fy;
    for (let col = 0; col < steps; col++) {
      const u = (left + (col + 0.5) * (right - left) / steps - cx) / fx;
      const [rayX, rayY] = undistortedRay(u, v, cam.dist);
      const target = origin.map((z, i) => z + reach * (rayX * rightAxis[i] + rayY * downAxis[i] + forwardAxis[i]));
      const bodyHit = intersection(origin, target, body);
      if (bodyHit === null) continue;
      hits++;
      if (solids.every(solid => {
        const hit = intersection(origin, target, solid);
        return hit === null || hit >= bodyHit - 1e-5;
      })) visible++;
    }
    const remaining = (steps - row - 1) * steps;
    // Clear views can finish early once the unseen rows cannot change the outcome.
    if (visible > 0 && visible >= minVisible * (hits + remaining)) return true;
  }
  return false;
}

// Body visibility is independent of the floor mask. Aim stays within 5 cm sideways.
export function personRays(cameras,selected,people,solids,{reachableOnly=false}={}){
  if(!selected)return [];
  const cylinder=p=>({kind:'cylinder',center:[p.x,p.y,0],radius:.25,height:1.8,id:p.id});
  const occluders=[...solids,...people.filter(p=>p!==selected).map(cylinder)];
  return cameras.flatMap(cam=>{
    const body=cylinder(selected);
    let blockedRay=null;
    function evaluate(target,offset,height,allowAim=true){
      if(!inFrame(cam,target))return null;
      const hit=firstHit(cam.center,target,occluders);
      const ray={camera:cam,person:selected,target,hit,end:hit?hit.point:target,offset,height};
      if(hit){if(allowAim)blockedRay??=ray;return null;}return ray;
    }
    const dx=cam.center[0]-selected.x,dy=cam.center[1]-selected.y,distance=Math.hypot(dx,dy);
    const nx=distance>1e-8?dx/distance:1,ny=distance>1e-8?dy/distance:0;
    const visibleArc=distance>.25?Math.acos(.25/distance):Math.PI/2,aimArc=Math.min(visibleArc,Math.asin(.05/.25));
    const heights=[.9,1.05,.75,1.2,.6,1.35,.45,1.5,.3,1.65,.15,1.79,.01,1.8,0];
    function targetAt(height,angle=null){
      const candidate=angle===null?[selected.x,selected.y,height]:[selected.x+.25*(nx*Math.cos(angle)-ny*Math.sin(angle)),selected.y+.25*(ny*Math.cos(angle)+nx*Math.sin(angle)),height];
      return firstHit(cam.center,candidate,[body])?.point||candidate;
    }
    for(const height of heights){
      const direct=evaluate(targetAt(height),0,height);if(direct)return [direct];
      const steps=height===.9?32:8;
      for(let i=1;i<=steps;i++)for(const sign of [1,-1]){
        const fraction=i/steps*.999999,ray=evaluate(targetAt(height,sign*fraction*aimArc),sign*fraction,height);
        if(ray)return [ray];
      }
    }
    if(reachableOnly)return [];
    // A visible outer edge still qualifies the person for a ray, but must not move
    // the aim beyond 5 cm. Keep the red central ray stopping at its first blocker.
    if(blockedRay)for(const height of heights)for(let i=1;i<=16;i++)for(const sign of [1,-1]){
      const angle=sign*i/16*.999999*visibleArc;if(Math.abs(angle)<=aimArc)continue;
      const visible=evaluate(targetAt(height,angle),sign*i/16,height,false);
      if(visible)return [{...blockedRay,visiblePoint:visible.target}];
    }
    return [];
  });
}
// A hypothetical standing person, or the existing person occupying this footprint.
export function personBlindAt(x,y,cameras,people,solids){
  let selected={id:'probe',x,y},nearest=.25;
  for(const p of people){const d=Math.hypot(p.x-x,p.y-y);if(d<=nearest){nearest=d;selected=p;}}
  const others=[...solids,...people.filter(p=>p!==selected).map(p=>({kind:'cylinder',center:[p.x,p.y,0],radius:.25,height:1.8}))];
  if(firstHit([selected.x,selected.y,.001],[selected.x,selected.y,.002],others))return false;
  return !personRays(cameras,selected,people,solids,{reachableOnly:true}).length;
}
export function insidePolygon(p,points){let yes=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])yes=!yes;}return yes;}

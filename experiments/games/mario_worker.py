"""Persistent JSON-lines NES worker. Run with .venv-games/bin/python."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore',category=DeprecationWarning)
import gym_super_mario_bros
from PIL import Image
ACTIONS={'wait':0,'right':128,'left':64,'jump':1,'jump_right':129,'jump_left':65}
class Mario:
 def reset(self,level='1-1',seed=0,record=None):
  if hasattr(self,'env'):self.env.close()
  self.env=gym_super_mario_bros.make(f'SuperMarioBros-{level}-v0',render_mode='rgb_array').unwrapped
  self.image,self.info=self.env.reset(seed=seed)
  self.frame=0;self.done=False;self.previous='wait';self.max_x=self.info['x_pos'];self.record=record;self.images=[]
  return self.observe()
 def observe(self):
  r=self.env.ram;x=int(self.info['x_pos']);foot=int(r[0xce])+32
  # The collision map is two alternating 16-column pages, 13 rows from y=32.
  # Report only terrain in the current viewport, not undisplayed future pages.
  camera=int(r[0x71a])*256+int(r[0x71c]);limit=min(x+160,camera+255)
  cols=[]
  for col in range(max(0,(x-16)//16),limit//16):
   vals=[int(r[0x500+(col//16%2)*208+row*16+col%16]) for row in range(13)]
   # Coin tiles 0xc2/0xc3 are collectible, not solid collision obstacles.
   solids=[i for i,v in enumerate(vals) if v and v not in (0xc2,0xc3)]
   segments=[]
   for row in solids:
    if segments and segments[-1][1]==32+row*16:segments[-1][1]+=16
    else:segments.append([32+row*16,48+row*16])
   cols.append({'x':col*16,'segments':segments})
  rects=[]
  for c in cols:
   for y1,y2 in c['segments']:
    match=next((p for p in reversed(rects) if p['x_end']==c['x'] and p['top']==y1 and p['bottom']==y2),None)
    if match:match['x_end']+=16
    else:rects.append({'x_start':c['x'],'x_end':c['x']+16,'top':y1,'bottom':y2})
  enemies=[]
  for i in range(5):
   if int(r[0xf+i]):
    ex=int(r[0x6e+i])*256+int(r[0x87+i]);ty=int(r[0x16+i]);ey=int(r[0x4b1+4*i])
    if camera<=ex<=camera+255 and ty<0x15:
     enemies.append({'type':{6:'goomba',0:'green koopa',1:'red koopa',13:'piranha plant'}.get(ty,f'enemy type {ty}'),'x':ex,'top':ey,'dx':ex-x})
  signed=lambda v:v if v<128 else v-256
  data={'x':x,'foot_y':foot,'grounded':int(r[0x1d])==0,'vertical_velocity':signed(int(r[0x9f])),
        'horizontal_velocity':round(signed(int(r[0x57]))/16,2),'previous':self.previous,'terrain':rects,'enemies':enemies}
  terrain='; '.join(f"x={p['x_start']}..{p['x_end']}, y={p['top']}..{p['bottom']}" for p in rects) or 'none'
  enemy='; '.join(f"{p['type']} at x={p['x']}, collision_top_y={p['top']}, horizontal offset={p['dx']}" for p in enemies) or 'none'
  state=(f"Mario x={x}, feet_y={foot}, size={self.info['status']}, grounded={data['grounded']}, horizontal velocity={data['horizontal_velocity']} pixels/frame, vertical velocity={data['vertical_velocity']} pixels/frame. Previous buttons: {self.previous}.\n"
         f"Visible solid terrain rectangles (world pixels): {terrain}. Empty space between rectangles is air, including pits where no floor exists.\nVisible enemies: {enemy}.\nFrame {self.frame}; furthest x reached={self.max_x}.")
  return {'state':state,'data':data,'info':self.info,'frame':self.frame,'done':self.done,'max_x':self.max_x}
 def step(self,action,frames=12):
  self.previous=action
  for _ in range(frames):
   if self.done:break
   self.image,_,term,trunc,self.info=self.env.step(ACTIONS[action]);self.done=term or trunc;self.frame+=1
   self.max_x=max(self.max_x,self.info['x_pos'])
   if self.record and self.frame%6==0:self.images.append(Image.fromarray(self.image.copy()))
  return self.observe()
 def save(self):
  if self.record and self.images:
   Path(self.record).parent.mkdir(parents=True,exist_ok=True)
   self.images[0].save(self.record,save_all=True,append_images=self.images[1:],duration=100,loop=0)
  return {'saved':self.record,'frames':len(self.images)}
 def reference(self):
  d=self.observe()['data'];x=d['x'];foot=d['foot_y']
  if not d['grounded']:return 'jump_right' if d['vertical_velocity']<0 else 'right'
  hazards=any(0<e['dx']<70 for e in d['enemies'])
  hazards|=any(0<p['x_start']-x<65 and p['top']<foot and p['bottom']>foot-32 for p in d['terrain'])
  for dx in (24,40,56):
   if not any(p['x_start']<=x+dx<p['x_end'] and foot<=p['top']<=foot+20 for p in d['terrain']):hazards=True
  # Need a release between distinct jumps, which the previous falling phase gives.
  return 'jump_right' if hazards and 'jump' not in self.previous else 'right'
if __name__=='__main__':
 m=Mario()
 for line in sys.stdin:
  try:
   q=json.loads(line);cmd=q.pop('cmd')
   if cmd=='reference':res={'choice':m.reference()}
   else:res=getattr(m,cmd)(**q)
   print(json.dumps(res),flush=True)
  except Exception as e:print(json.dumps({'error':repr(e)}),flush=True)

"""Deterministic normal-training examples for ROI geometry inspection."""
from PIL import Image,ImageDraw
from common import ROOT,TASKS,read,dump

def main():
    videos=read(ROOT/'feature-index.json');tiles=[];records=[]
    for task in TASKS:
        v=next(v for v in videos if v['task']==task and v['split']=='train')
        det=read(ROOT/'roi'/(v['video']+'.json'))
        for index in [0,len(v['rows'])//2]:
            r=v['rows'][index];uid=r['frame_uids'][1]
            d=next(d for d in det if d['uid']==uid and d['names']==r['objects'])
            im=Image.open(ROOT/'cache/tail-images'/(uid+'.jpg')).convert('RGB')
            draw=ImageDraw.Draw(im)
            for slot,b in enumerate(d['selected']):
                if b:draw.rectangle(b['box'],outline=['red','cyan'][slot],width=4)
            im.thumbnail((640,360));tile=Image.new('RGB',(660,435),'white');tile.paste(im,(10,65));draw=ImageDraw.Draw(tile)
            draw.text((10,5),task+' normal train, step '+str(r['step']),fill='black')
            for slot,b in enumerate(d['selected']):
                draw.text((10,23+17*slot),r['objects'][slot]+': '+('fallback' if b is None else b['label']+' '+str(round(b['score'],3))),fill=['red','blue'][slot])
            tiles.append(tile);records.append({'id':r['id'],'uid':uid,'names':r['objects'],'selected':d['selected']})
    sheet=Image.new('RGB',(1320,870),'white')
    for i,t in enumerate(tiles):sheet.paste(t,((i%2)*660,(i//2)*435))
    sheet.save(ROOT/'follow-up/normal-roi-qa.jpg',quality=94)
    dump(ROOT/'follow-up/normal-roi-qa.json',records)

if __name__=='__main__':main()

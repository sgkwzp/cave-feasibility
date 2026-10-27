"""Normal-only visual geometry QA for each added task."""
from PIL import Image,ImageDraw
from common import ROOT,read,dump

EXT=ROOT/'remaining-tasks'
def main():
    cfg=read(EXT/'config.json');index=read(EXT/'feature-index.json');tiles=[];records=[]
    for task in cfg['tasks']:
        v=next(v for v in index if v['task']==task and v['split']=='train')
        detections=read(EXT/'roi'/(v['video']+'.json'))
        for i in [0,len(v['rows'])//2]:
            row=v['rows'][i];uid=row['frame_uids'][1]
            d=next(d for d in detections if d['uid']==uid and d['names']==row['objects'])
            im=Image.open(EXT/'cache/tail-images'/(uid+'.jpg')).convert('RGB');draw=ImageDraw.Draw(im)
            for k,b in enumerate(d['selected']):
                if b:draw.rectangle(b['box'],outline=['red','cyan'][k],width=4)
            im.thumbnail((640,360));tile=Image.new('RGB',(660,430),'white');tile.paste(im,(10,65));draw=ImageDraw.Draw(tile)
            draw.text((10,5),task+' normal train, step '+str(row['step']),fill='black')
            for k,b in enumerate(d['selected']):draw.text((10,23+17*k),row['objects'][k]+': '+('fallback' if b is None else b['label']+' '+str(round(b['score'],3))),fill=['red','blue'][k])
            tiles.append(tile);records.append({'id':row['id'],'uid':uid,'names':row['objects'],'selected':d['selected']})
    sheet=Image.new('RGB',(1320,1290),'white')
    for i,t in enumerate(tiles):sheet.paste(t,((i%2)*660,(i//2)*430))
    sheet.save(EXT/'normal-roi-qa.jpg',quality=94);dump(EXT/'normal-roi-qa.json',records)
    print('NORMAL_ROI_QA',len(records),flush=True)

if __name__=='__main__':main()

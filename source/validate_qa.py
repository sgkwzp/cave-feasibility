"""Normal-training frame geometry review only; no semantic test annotations."""
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,read,dump
OUT=ROOT/'deployment-mechanism'

def main():
    cfg=read(OUT/'config.json');canvas=Image.new('RGB',(1440,5*340),'white');draw=ImageDraw.Draw(canvas);records=[]
    originals={v['video']:(base,v) for base in [ROOT,ROOT/'remaining-tasks'] for v in read(base/'feature-index.json')}
    for ti,task in enumerate(cfg['tasks']):
        v=next(v for v in read(OUT/'manifest.json') if v['task']==task and v['split']=='train')
        if not (OUT/'observations'/(v['video']+'.json')).exists():continue
        meta=read(OUT/'observations'/(v['video']+'.json'));row=next(r for r in meta['rows'] if r['arm']=='Pre')
        base,old=originals[v['video']];post=next(r for r in old['rows'] if r['id']==row['source_id'])
        pred=next(r for r in meta['rows'] if r['arm']=='Pred' and r['step']==row['step'])
        images={e['uid']:e['image'] for e in meta['entries']};pred_det={(d['uid'],tuple(d['names'])):d for d in meta['detections']}
        old_det={(d['uid'],tuple(d['names'])):d for d in read(base/'roi'/(v['video']+'.json'))}
        y=ti*340;draw.text((5,y+3),f'{task} | {v["video"]} | GT step {row["step"]} | normal train',fill='black')
        for j,(name,r,uid,det,path) in enumerate([
            ('Pre',row,row['frame_uids'][1],pred_det,images[row['frame_uids'][1]]),
            ('Post',post,post['frame_uids'][1],old_det,str(base/'cache/tail-images'/(post['frame_uids'][1]+'.jpg'))),
            ('Predicted tail',pred,pred['frame_uids'][1],pred_det,images[pred['frame_uids'][1]])]):
            im=Image.open(path).convert('RGB');d=ImageDraw.Draw(im);entry=det[(uid,tuple(r['objects']))]
            for k,b in enumerate(entry['selected']):
                if b is not None:d.rectangle(b['box'],outline=['red','lime'][k],width=3)
            im.thumbnail((470,265));x=j*480+5;canvas.paste(im,(x,y+50))
            draw.text((x,y+24),name+' | '+' / '.join(r['objects']),fill='black')
            records.append({'task':task,'video':v['video'],'arm':name,'id':r['id'],'uid':uid,'image':path})
    canvas.save(OUT/'normal-window-qa.jpg',quality=93);dump(OUT/'normal-window-qa.json',records)

if __name__=='__main__':main()

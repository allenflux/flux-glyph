import copy
import json
from pathlib import Path
import tempfile
import unittest

from training.capture import capture_android as capture


class AndroidCaptureMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.font_path=self.root/'font.ttf';self.font_path.write_bytes(b'font fixture, native shaper is mocked')
        self.font={'id':'font','family':'Noto Sans CJK SC','training_family':'Noto Sans CJK SC','postscript':'FixturePS',
            'path':str(self.font_path),'sha256':capture.sha(self.font_path),'ttc_index':0,'split':'all'}
        self.requested={'id':'page-r0','font_id':'font','text':'Test','script':'latin','font_size_px':20,'color':'#333333','bbox':[0,0,100,50]}
        self.page={'id':'page','split':'train','background':'#FFFFFF','regions':[self.requested]}
        self.scenes={'schema':'flux-glyph-android-scenes-v1','canvas_px':[1080,2400],
            'families':['Noto Sans CJK SC']+[f'Family{i}' for i in range(8)]+['__unknown__'],
            'fonts':[self.font],'pages':[self.page],'bindings':{}}

    def test_scene_rejects_cross_partition_normalized_text_and_source_groups(self):
        for change in ('text','group'):
            scenes=copy.deepcopy(self.scenes);page=copy.deepcopy(self.page);page.update(id='page2',split='test')
            page['regions'][0].update(id='page2-r0',text='ＴＥＳＴ' if change=='text' else 'Different')
            if change=='group':scenes['pages'][0]['content_group_id']=page['content_group_id']='shared'
            scenes['pages'].append(page)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'crosses partitions'):
                capture.validate_scenes(scenes)

    def test_scene_rejects_invalid_style_or_font_alias(self):
        for key,value in [('font_size_px',float('nan')),('color','#GG0000'),('bbox',[0.,0,100,50])]:
            scenes=copy.deepcopy(self.scenes);scenes['pages'][0]['regions'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):capture.validate_scenes(scenes)
        scenes=copy.deepcopy(self.scenes);scenes['fonts'].append({**self.font,'id':'alias','training_family':'Family0'})
        with self.assertRaisesRegex(ValueError,'conflicting training classes'):capture.validate_scenes(scenes)

    def test_unknown_family_and_source_cannot_cross_splits(self):
        scenes=copy.deepcopy(self.scenes)
        first={**self.font,'id':'unknown1','training_family':'__unknown__','family':'Unknown','split':'train'}
        second={**first,'id':'unknown2','split':'test'}
        scenes['fonts']=[first,second]
        with self.assertRaisesRegex(ValueError,'unknown font family/source crosses'):capture.validate_scenes(scenes)

    def label(self):
        return {'schema':'flux-glyph-android-native-region-v1','split':'train','source_id':'android:page','page_id':'page',
            'content_group_id':'page','region_id':'page-r0','image':'screenshots/page.png','proof':'proofs/page.json',
            'image_width':1080,'image_height':2400,'bbox':[0,0,100,50],'ink_bbox':[0,2,40,22],'text':'Test','script':'latin',
            'font_id':'font','font_family':self.font['family'],'training_family':self.font['training_family'],'font_face':'FixturePS',
            'font_file_sha256':self.font['sha256'],'ttc_index':0,'font_size_screen_px':20,'color':'#333333','background':'#FFFFFF',
            'native_font_verified':True,'native_rendering':'Android Canvas + TextRunShaper'}

    def test_label_metadata_binds_ttc_group_script_background_and_dimensions(self):
        row=self.label();capture.verify_label_metadata(row,self.page,self.requested,self.font,[1080,2400])
        for key,value in [('ttc_index',1),('content_group_id','other'),('script','han'),('background','#000000'),('image_width',100)]:
            changed={**row,key:value}
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'label metadata differs'):
                capture.verify_label_metadata(changed,self.page,self.requested,self.font,[1080,2400])

    def test_region_rechecks_pixels_even_when_native_proof_is_cached(self):
        row=self.label();(self.root/'screenshots').mkdir();(self.root/'proofs').mkdir()
        image_path=self.root/row['image'];image_path.write_bytes(b'native framebuffer fixture');row['image_sha256']=capture.sha(image_path)
        glyph={'glyph_id':1,'font_verified':True,'font':{'sha256':self.font['sha256'],'postscript':'FixturePS','ttc_index':0}}
        actual={'id':row['region_id'],'font_id':'font','text':'Test','bbox':row['bbox'],'ink_bbox':row['ink_bbox'],
            'font_size_px':20,'color':'#333333','font_verified':True,'glyphs':[glyph]}
        proof={'canvas_px':[1080,2400],'build_fingerprint':'fixture','sdk_int':35,'page_id':'page','split':'train','regions':[actual]}
        capture.dump(self.root/row['proof'],proof)
        cache={'rows':{row['region_id']:row},'index':{row['region_id']:(self.page,self.requested)},'fonts':{'font':self.font},
            'files':{row['image']:row['image_sha256'],row['proof']:capture.sha(self.root/row['proof'])},'proofs':{},
            'manifest':{'device':{'build_fingerprint':'fixture','sdk_int':35}}}
        capture._CACHE[str(self.root.resolve())]=cache
        self.addCleanup(capture._CACHE.pop,str(self.root.resolve()),None)
        capture.verify_region(self.root,row)
        image_path.write_bytes(b'changed framebuffer')
        with self.assertRaisesRegex(ValueError,'native screenshot changed'):capture.verify_region(self.root,row)


if __name__=='__main__':unittest.main()
